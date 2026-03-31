/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.MultiSegmentCagraSearch;
import com.nvidia.cuvs.MultiSegmentSearchResults;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;

/**
 * Extends {@link KnnFloatVectorQuery} for GPU-only search.
 *
 * <p>When all index segments use {@link CuVS2510GPUVectorsReader} and the query uses CAGRA
 * (k&nbsp;&le;&nbsp;1024, no explicit filter), {@link #rewrite} runs a globally-optimized
 * multi-segment search:
 * <ol>
 *   <li>All per-segment CAGRA searches write into a single shared device buffer without any
 *       per-segment device-to-host copy or stream synchronization.</li>
 *   <li>A single {@code cuvsSelectK} call finds the global top-k entirely on GPU.</li>
 *   <li>Results are copied to host in one pass and mapped to Lucene doc IDs.</li>
 * </ol>
 *
 * <p>Falls back to the standard per-segment Lucene path when the optimized path cannot be
 * applied (mixed segment types, explicit query filter, k&nbsp;&gt;&nbsp;1024, brute-force
 * fallback needed, or missing CAGRA index).
 *
 * @since 25.10
 */
public class GPUKnnFloatVectorQuery extends KnnFloatVectorQuery {

  private final int iTopK;
  private final int searchWidth;

  /**
   * Initializes {@link GPUKnnFloatVectorQuery}.
   *
   * @param field       the vector field name
   * @param target      the query vector
   * @param k           the number of nearest neighbors to return
   * @param filter      optional pre-filter query
   * @param iTopK       CAGRA itopk_size parameter
   * @param searchWidth CAGRA search_width parameter
   */
  public GPUKnnFloatVectorQuery(
      String field, float[] target, int k, Query filter, int iTopK, int searchWidth) {
    super(field, target, k, filter);
    this.iTopK = iTopK;
    this.searchWidth = searchWidth;
  }

  // -------------------------------------------------------------------------
  // Optimized multi-segment path
  // -------------------------------------------------------------------------

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    // Only apply the optimized path when there is no explicit filter.
    // Live-document filtering (deletions) is handled via acceptDocs below.
    if (filter != null) {
      return super.rewrite(indexSearcher);
    }
    // CAGRA search is capped at k=1024.
    if (k > 1024) {
      return super.rewrite(indexSearcher);
    }

    IndexReader reader = indexSearcher.getIndexReader();
    List<LeafReaderContext> leaves = reader.leaves();
    if (leaves.isEmpty()) {
      return new MatchNoDocsQuery();
    }

    // Collect a CuVS2510GPUVectorsReader for every segment; fall back if any segment
    // lacks one or has no CAGRA index for this field.
    List<CuVS2510GPUVectorsReader> gpuReaders = new ArrayList<>(leaves.size());
    for (LeafReaderContext ctx : leaves) {
      CuVS2510GPUVectorsReader gpuReader = unwrapGpuReader(ctx);
      if (gpuReader == null || gpuReader.getCagraIndexForField(field) == null) {
        return super.rewrite(indexSearcher);
      }
      gpuReaders.add(gpuReader);
    }

    // Build one CagraIndex + CagraQuery per segment.
    CuVSResources resources = getCuVSResourcesInstance();
    List<CagraIndex> cagraIndices = new ArrayList<>(leaves.size());
    List<CagraQuery> cagraQueries = new ArrayList<>(leaves.size());

    try {
      float[] target = getTargetCopy();
      CagraSearchParams searchParams =
          new CagraSearchParams.Builder()
              .withItopkSize(Math.max(iTopK, k))
              .withSearchWidth(searchWidth)
              .build();

      for (int i = 0; i < leaves.size(); i++) {
        LeafReaderContext ctx = leaves.get(i);
        cagraIndices.add(gpuReaders.get(i).getCagraIndexForField(field));

        // Pass live-document bits as the prefilter so deleted docs are excluded.
        Bits liveDocs = ctx.reader().getLiveDocs();
        CagraQuery query =
            buildCagraQuery(resources, target, k, searchParams, liveDocs, gpuReaders.get(i), ctx);
        cagraQueries.add(query);
      }

      MultiSegmentSearchResults results =
          MultiSegmentCagraSearch.search(resources, cagraIndices, cagraQueries, k);

      if (results.count() == 0) {
        return new MatchNoDocsQuery();
      }

      // Map (segmentIdx, ordinal) → global Lucene doc ID; compute normalized score.
      ScoreDoc[] scoreDocs = new ScoreDoc[results.count()];
      for (int j = 0; j < results.count(); j++) {
        int segIdx = results.getSegmentIndex(j);
        int ordinal = results.getOrdinal(j);
        float dist = results.getDistance(j);

        LeafReaderContext ctx = leaves.get(segIdx);
        int localDoc = gpuReaders.get(segIdx).ordToDoc(field, ordinal);
        int globalDoc = ctx.docBase + localDoc;
        float score = 1.0f / (1.0f + dist);
        scoreDocs[j] = new ScoreDoc(globalDoc, score);
      }

      Arrays.sort(scoreDocs, Comparator.comparingDouble((ScoreDoc sd) -> sd.score).reversed());
      return docAndScoreQuery(scoreDocs);

    } catch (Throwable t) {
      if (t instanceof IOException) throw (IOException) t;
      if (t instanceof RuntimeException) throw (RuntimeException) t;
      throw new RuntimeException("Multi-segment GPU search failed", t);
    }
  }

  // -------------------------------------------------------------------------
  // Per-segment fallback path (used when filter != null or k > 1024)
  // -------------------------------------------------------------------------

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Bits acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {
    GPUPerLeafCuVSKnnCollector results =
        new GPUPerLeafCuVSKnnCollector(k, visitedLimit, iTopK, searchWidth);
    context.reader().searchNearestVectors(field, getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  /**
   * Unwraps the {@link LeafReaderContext}'s reader to a {@link CuVS2510GPUVectorsReader}, or
   * returns {@code null} if the reader is not of that type.
   */
  private static CuVS2510GPUVectorsReader unwrapGpuReader(LeafReaderContext ctx) {
    var unwrapped = FilterLeafReader.unwrap(ctx.reader());
    if (!(unwrapped instanceof CodecReader)) return null;
    KnnVectorsReader vr = ((CodecReader) unwrapped).getVectorReader();
    return (vr instanceof CuVS2510GPUVectorsReader gpuReader) ? gpuReader : null;
  }

  /**
   * Builds a {@link CagraQuery} for a single segment, incorporating live-document filtering
   * via a prefilter bitset when deletions are present.
   */
  private CagraQuery buildCagraQuery(
      CuVSResources resources,
      float[] target,
      int topK,
      CagraSearchParams searchParams,
      Bits liveDocs,
      CuVS2510GPUVectorsReader gpuReader,
      LeafReaderContext ctx)
      throws IOException {
    CuVSMatrix.Builder<?> vectorBuilder =
        CuVSMatrix.deviceBuilder(resources, 1, target.length, CuVSMatrix.DataType.FLOAT);
    vectorBuilder.addVector(target);
    CuVSMatrix queryVector = vectorBuilder.build();

    CagraQuery.Builder queryBuilder =
        new CagraQuery.Builder(resources)
            .withTopK(topK)
            .withSearchParams(searchParams)
            .withQueryVectors(queryVector);

    if (liveDocs != null) {
      // Convert liveDocs to a BitSet over vector ordinals so CAGRA can filter on GPU.
      var rawValues = gpuReader.getFloatVectorValues(field);
      Bits acceptedOrds = rawValues.getAcceptOrds(liveDocs);
      int length = acceptedOrds.length();
      BitSet mask = new BitSet(length);
      for (int i = 0; i < length; i++) {
        if (acceptedOrds.get(i)) mask.set(i);
      }
      queryBuilder.withPrefilter(mask, length);
    }

    return queryBuilder.build();
  }

  /**
   * Builds a {@link Query} that matches exactly the given pre-scored documents.
   *
   * <p>Partitions {@code scoreDocs} by segment (using {@link ScoreDoc#shardIndex} as the segment
   * offset relative to {@link LeafReaderContext#docBase}), then returns a {@link Scorer} per
   * segment that iterates those docs in ascending doc-ID order and replays their pre-computed
   * scores.
   */
  private static Query docAndScoreQuery(ScoreDoc[] scoreDocs) {
    return new Query() {
      @Override
      public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
          throws IOException {
        return new Weight(this) {
          @Override
          public ScorerSupplier scorerSupplier(LeafReaderContext ctx) {
            int base = ctx.docBase;
            int maxDoc = base + ctx.reader().maxDoc();
            // Collect docs belonging to this segment; re-sort by local doc ID ascending.
            int[] localDocs = new int[scoreDocs.length];
            float[] localScores = new float[scoreDocs.length];
            int count = 0;
            for (ScoreDoc sd : scoreDocs) {
              if (sd.doc >= base && sd.doc < maxDoc) {
                localDocs[count] = sd.doc - base;
                localScores[count] = sd.score * boost;
                count++;
              }
            }
            if (count == 0) return null;
            final int n = count;
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Arrays.sort(idx, Comparator.comparingInt(i -> localDocs[i]));
            final int[] sortedDocs = new int[n];
            final float[] sortedScores = new float[n];
            for (int i = 0; i < n; i++) {
              sortedDocs[i] = localDocs[idx[i]];
              sortedScores[i] = localScores[idx[i]];
            }

            return new ScorerSupplier() {
              @Override
              public Scorer get(long leadCost) {
                return new Scorer() {
                  private int pos = -1;

                  @Override
                  public DocIdSetIterator iterator() {
                    return new DocIdSetIterator() {
                      @Override
                      public int docID() {
                        return pos < 0 ? -1 : (pos >= n ? NO_MORE_DOCS : sortedDocs[pos]);
                      }

                      @Override
                      public int nextDoc() {
                        pos++;
                        return docID();
                      }

                      @Override
                      public int advance(int target) {
                        while (pos < n && sortedDocs[pos] < target) pos++;
                        return docID();
                      }

                      @Override
                      public long cost() {
                        return n;
                      }
                    };
                  }

                  @Override
                  public float getMaxScore(int upTo) {
                    return Float.MAX_VALUE;
                  }

                  @Override
                  public float score() {
                    return sortedScores[pos];
                  }

                  @Override
                  public int docID() {
                    return pos < 0
                        ? -1
                        : (pos >= n ? DocIdSetIterator.NO_MORE_DOCS : sortedDocs[pos]);
                  }
                };
              }

              @Override
              public long cost() {
                return n;
              }
            };
          }

          @Override
          public boolean isCacheable(LeafReaderContext ctx) {
            return false;
          }

          @Override
          public Explanation explain(LeafReaderContext ctx, int doc) {
            for (ScoreDoc sd : scoreDocs) {
              if (sd.doc == ctx.docBase + doc) {
                return Explanation.match(sd.score, "GPU multi-segment CAGRA search");
              }
            }
            return Explanation.noMatch("not a GPU search result");
          }
        };
      }

      @Override
      public String toString(String field) {
        return "GPUDocAndScoreQuery";
      }

      @Override
      public void visit(QueryVisitor visitor) {}

      @Override
      public boolean equals(Object o) {
        return this == o;
      }

      @Override
      public int hashCode() {
        return System.identityHashCode(this);
      }
    };
  }
}
