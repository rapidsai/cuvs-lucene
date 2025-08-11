/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_INDEX_EXT;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.VERSION_CURRENT;
import static com.nvidia.cuvs.lucene.CuVSVectorsReader.handleThrowable;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/** KnnVectorsWriter for CuVS, responsible for merge and flush of vectors into GPU */
public class CuVSVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED = shallowSizeOfInstance(CuVSVectorsWriter.class);

  @SuppressWarnings("unused")
  private static final Logger log = Logger.getLogger(CuVSVectorsWriter.class.getName());

  /** The name of the CUVS component for the info-stream * */
  public static final String CUVS_COMPONENT = "CUVS";

  // The minimum number of vectors in the dataset required before
  // we attempt to build a Cagra index
  static final int MIN_CAGRA_INDEX_SIZE = 2;

  private final int cuvsWriterThreads;
  private final int intGraphDegree;
  private final int graphDegree;

  private final CuVSResources resources;
  private final IndexType indexType;

  private final FlatVectorsWriter flatVectorsWriter; // for writing the raw vectors
  private final List<CuVSFieldWriter> fields = new ArrayList<>();
  private final IndexOutput meta, cuvsIndex;
  private final InfoStream infoStream;
  private boolean finished;

  /** The CuVS index Type. */
  public enum IndexType {
    /** Builds a Cagra index. */
    CAGRA(true, false, false),
    /** Builds a Brute Force index. */
    BRUTE_FORCE(false, true, false),
    /** Builds an HSNW index - suitable for searching on CPU. */
    HNSW(false, false, true),
    /** Builds a Cagra and a Brute Force index. */
    CAGRA_AND_BRUTE_FORCE(true, true, false);
    private final boolean cagra, bruteForce, hnsw;

    IndexType(boolean cagra, boolean bruteForce, boolean hnsw) {
      this.cagra = cagra;
      this.bruteForce = bruteForce;
      this.hnsw = hnsw;
    }

    public boolean cagra() {
      return cagra;
    }

    public boolean bruteForce() {
      return bruteForce;
    }

    public boolean hnsw() {
      return hnsw;
    }
  }

  public CuVSVectorsWriter(
      SegmentWriteState state,
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      IndexType indexType,
      CuVSResources resources,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.indexType = indexType;
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.resources = resources;
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, CUVS_META_CODEC_EXT);
    String cagraFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, CUVS_INDEX_EXT);

    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      cuvsIndex = state.directory.createOutput(cagraFileName, state.context);
      CodecUtil.writeIndexHeader(
          meta,
          CUVS_META_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          cuvsIndex,
          CUVS_INDEX_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    @SuppressWarnings("unchecked")
    var flatWriter = (FlatFieldVectorsWriter<float[]>) writer;
    var cuvsFieldWriter = new CuVSFieldWriter(fieldInfo, flatWriter);
    fields.add(cuvsFieldWriter);
    return cuvsFieldWriter;
  }

  static String indexMsg(int size, int... args) {
    StringBuilder sb = new StringBuilder("cagra index params");
    sb.append(": size=").append(size);
    sb.append(", intGraphDegree=").append(args[0]);
    sb.append(", actualIntGraphDegree=").append(args[1]);
    sb.append(", graphDegree=").append(args[2]);
    sb.append(", actualGraphDegree=").append(args[3]);
    return sb.toString();
  }

  private CagraIndexParams cagraIndexParams(int size) {
    if (size < 2) {
      // https://github.com/rapidsai/cuvs/issues/666
      throw new IllegalArgumentException("cagra index must be greater than 2");
    }

    return new CagraIndexParams.Builder()
        .withNumWriterThreads(cuvsWriterThreads)
        .withIntermediateGraphDegree(intGraphDegree)
        .withGraphDegree(graphDegree)
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .build();
  }

  static long nanosToMillis(long nanos) {
    return Duration.ofNanos(nanos).toMillis();
  }

  private void info(String msg) {
    if (infoStream.isEnabled(CUVS_COMPONENT)) {
      infoStream.message(CUVS_COMPONENT, msg);
    }
  }

  private void writeCagraIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams params = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    var index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();
    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info("Cagra index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    Path tmpFile = Files.createTempFile(resources.tempDirectory(), "tmpindex", "cag");
    index.serialize(os, tmpFile);
    index.destroyIndex();
  }

  private void writeBruteForceIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    BruteForceIndexParams params =
        new BruteForceIndexParams.Builder().withNumWriterThreads(cuvsWriterThreads).build();
    long startTime = System.nanoTime();
    var index =
        BruteForceIndex.newBuilder(resources).withIndexParams(params).withDataset(dataset).build();
    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info("bf index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    index.serialize(os);
    index.destroyIndex();
  }

  private void writeHNSWIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams indexParams = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    var index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(indexParams).build();
    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info("HNSW index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    Path tmpFile = Files.createTempFile("tmpindex", "hnsw");
    index.serializeToHNSW(os, tmpFile);
    index.destroyIndex();
  }

  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    flatVectorsWriter.flush(maxDoc, sortMap);
    for (var field : fields) {
      if (sortMap == null) {
        writeField(field);
      } else {
        writeSortingField(field, sortMap);
      }
    }
  }

  private void writeField(CuVSFieldWriter fieldData) throws IOException {
    List<float[]> vectors = fieldData.getVectors();
    CuVSMatrix dataset = createMatrixFromVectorList(vectors);
    if (dataset == null) {
      writeEmpty(fieldData.fieldInfo());
      return;
    }
    writeFieldInternal(fieldData.fieldInfo(), dataset);
  }

  private void writeSortingField(CuVSFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {
    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord

    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);

    List<float[]> oldVectors = fieldData.getVectors();
    if (oldVectors.isEmpty()) {
      writeEmpty(fieldData.fieldInfo());
      return;
    }

    int vectorCount = oldVectors.size();

    // Create sorted array directly with pre-allocated size
    float[][] sortedVectors = new float[vectorCount][];
    for (int i = 0; i < vectorCount; i++) {
      sortedVectors[i] = oldVectors.get(new2OldOrd[i]);
    }

    CuVSMatrix dataset = CuVSMatrix.ofArray(sortedVectors);
    writeFieldInternal(fieldData.fieldInfo(), dataset);
  }

  private void writeFieldInternal(FieldInfo fieldInfo, CuVSMatrix dataset) throws IOException {
    if (dataset.size() == 0) {
      writeEmpty(fieldInfo);
      return;
    }
    long cagraIndexOffset, cagraIndexLength = 0L;
    long bruteForceIndexOffset, bruteForceIndexLength = 0L;
    long hnswIndexOffset, hnswIndexLength = 0L;

    // workaround for the minimum number of vectors for Cagra
    IndexType indexType =
        this.indexType.cagra() && dataset.size() < MIN_CAGRA_INDEX_SIZE
            ? IndexType.BRUTE_FORCE
            : this.indexType;

    try {
      cagraIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.cagra()) {
        try {
          var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
          writeCagraIndex(cagraIndexOutputStream, dataset);
        } catch (Throwable t) {
          handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
          // workaround for cuVS issue
          indexType = IndexType.BRUTE_FORCE;
        }
        cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;
      }

      bruteForceIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.bruteForce()) {
        var bruteForceIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
        writeBruteForceIndex(bruteForceIndexOutputStream, dataset);
        bruteForceIndexLength = cuvsIndex.getFilePointer() - bruteForceIndexOffset;
      }

      hnswIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.hnsw()) {
        var hnswIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
        if (dataset.size() > MIN_CAGRA_INDEX_SIZE) {
          try {
            writeHNSWIndex(hnswIndexOutputStream, dataset);
          } catch (Throwable t) {
            handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
          }
        }
        hnswIndexLength = cuvsIndex.getFilePointer() - hnswIndexOffset;
      }

      writeMeta(
          fieldInfo,
          (int) dataset.size(),
          cagraIndexOffset,
          cagraIndexLength,
          bruteForceIndexOffset,
          bruteForceIndexLength,
          hnswIndexOffset,
          hnswIndexLength);
    } catch (Throwable t) {
      handleThrowable(t);
    }
  }

  private void writeEmpty(FieldInfo fieldInfo) throws IOException {
    writeMeta(fieldInfo, 0, 0L, 0L, 0L, 0L, 0L, 0L);
  }

  private void writeMeta(
      FieldInfo field,
      int count,
      long cagraIndexOffset,
      long cagraIndexLength,
      long bruteForceIndexOffset,
      long bruteForceIndexLength,
      long hnswIndexOffset,
      long hnswIndexLength)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeInt(field.getVectorDimension());
    meta.writeInt(count);
    meta.writeVLong(cagraIndexOffset);
    meta.writeVLong(cagraIndexLength);
    meta.writeVLong(bruteForceIndexOffset);
    meta.writeVLong(bruteForceIndexLength);
    meta.writeVLong(hnswIndexOffset);
    meta.writeVLong(hnswIndexLength);
  }

  static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < SIMILARITY_FUNCTIONS.size(); i++) {
      if (SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  // We currently ignore this, until cuVS supports tiered indices
  private static final String CANNOT_GENERATE_CAGRA =
      """
      Could not generate an intermediate CAGRA graph because the initial \
      kNN graph contains too many invalid or duplicated neighbor nodes. \
      This error can occur, for example, if too many overflows occur \
      during the norm computation between the dataset vectors\
      """;

  static void handleThrowableWithIgnore(Throwable t, String msg) throws IOException {
    if (t.getMessage().contains(msg)) {
      return;
    }
    handleThrowable(t);
  }

  /** Creates CuVSMatrix directly from FloatVectorValues without intermediate List. */
  private CuVSMatrix getVectorDataMatrix(
      FloatVectorValues floatVectorValues, int vectorCount, int dimensions) throws IOException {
    if (vectorCount == 0) {
      return null;
    }

    // Pre-allocate exact size array instead of using ArrayList
    float[][] vectorArray = new float[vectorCount][];
    // Use ordinal-based mapping to ensure vectors are stored in the correct order
    // The ordinal from iter.index() should match the position that vectorValue(index) expects
    KnnVectorValues.DocIndexIterator iter = floatVectorValues.iterator();
    int rowIndex = 0;
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      float[] vector = floatVectorValues.vectorValue(iter.index());
      if (vector != null) {
        int ordinal = iter.index();
        // Store vector at the ordinal position to match vectorValue(index) expectations
        vectorArray[ordinal] = vector;
        rowIndex++;
      }
    }

    // Resize if needed (though vectorCount should be accurate)
    if (rowIndex < vectorCount) {
      float[][] resized = new float[rowIndex][];
      System.arraycopy(vectorArray, 0, resized, 0, rowIndex);
      vectorArray = resized;
    }

    // Handle empty case that CuVSMatrix.ofArray doesn't support
    if (vectorArray.length == 0) {
      return null;
    }

    return CuVSMatrix.ofArray(vectorArray);
  }

  /** Creates CuVSMatrix from List<float[]> efficiently with pre-allocated array. */
  private CuVSMatrix createMatrixFromVectorList(List<float[]> vectors) {
    if (vectors.isEmpty()) {
      return null;
    }

    // Convert to array more efficiently
    return CuVSMatrix.ofArray(vectors.toArray(new float[vectors.size()][]));
  }

  /** Creates CuVSMatrix from merged vectors, using dense array without gaps. */
  private CuVSMatrix createMatrixFromMergedVectors(
      FloatVectorValues mergedVectorValues, int vectorCount, int dimensions) throws IOException {
    if (vectorCount == 0) {
      return null;
    }

    // Use a dense array approach to avoid null elements that CuVSMatrix.ofArray can't handle
    List<float[]> vectorList = new ArrayList<>();

    KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      int ordinal = iter.index();
      float[] vector = mergedVectorValues.vectorValue(ordinal);
      if (vector != null) {
        // Clone the vector to ensure we have distinct arrays. This is necessary because
        // FloatVectorValues may reuse the same array instance for multiple vectors to
        // reduce allocations. Without cloning, all entries in our list would point to
        // the same array containing only the last vector's values.
        vectorList.add(vector.clone());
      }
    }

    if (vectorList.isEmpty()) {
      return null;
    }

    return CuVSMatrix.ofArray(vectorList.toArray(new float[vectorList.size()][]));
  }

  /** Legacy method that copies vector values into a 2D array. */
  private static float[][] getVectorDataArray(FloatVectorValues floatVectorValues, int expectedSize)
      throws IOException {
    List<float[]> vectorList = new ArrayList<>();
    KnnVectorValues.DocIndexIterator iter = floatVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      float[] vector = floatVectorValues.vectorValue(iter.index());
      if (vector != null) {
        vectorList.add(vector);
      }
    }
    return vectorList.toArray(new float[vectorList.size()][]);
  }

  /**
   * Merges CAGRA indexes using the native CuVS merge API instead of rebuilding from vectors.
   * Falls back to vector-based merge if native merge fails.
   */
  private void mergeCagraIndexes(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      // Collect existing CAGRA indexes from merge segments
      List<CagraIndex> cagraIndexes = new ArrayList<>();

      // Get total vector count from the merged vector values to be accurate
      final FloatVectorValues mergedVectorValues =
          switch (fieldInfo.getVectorEncoding()) {
            case BYTE -> throw new AssertionError("bytes not supported");
            case FLOAT32 ->
                KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
          };
      int totalVectorCount = mergedVectorValues.size();

      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        var knnReader = mergeState.knnVectorsReaders[i];
        if (knnReader instanceof CuVSVectorsReader cuvsReader) {
          // Access the CAGRA index for this field from the reader
          CagraIndex cagraIndex = getCagraIndexFromReader(cuvsReader, fieldInfo.name);
          if (cagraIndex != null) {
            cagraIndexes.add(cagraIndex);
          }
        }
      }

      if (cagraIndexes.size() > 1) {
        // Use native CAGRA merge API when we have multiple valid indexes
        CagraIndex mergedIndex = CagraIndex.merge(cagraIndexes.toArray(new CagraIndex[0]));
        writeMergedCagraIndex(fieldInfo, mergedIndex, totalVectorCount);
        info(
            "Successfully merged " + cagraIndexes.size() + " CAGRA indexes using native merge API");
      } else {
        // Fall back to vector-based approach for single/no indexes
        throw new RuntimeException("Only a single CAGRA index found");
      }
    } catch (Throwable t) {
      throw new RuntimeException(
          "Native CAGRA merge failed, falling back to vector-based merge: ", t);
    }
  }

  /**
   * Extracts the CAGRA index for a specific field from a CuVSVectorsReader.
   */
  private CagraIndex getCagraIndexFromReader(CuVSVectorsReader reader, String fieldName) {
    try {
      // Use reflection to access the private cuvsIndices field
      var cuvsIndicesField = reader.getClass().getDeclaredField("cuvsIndices");
      cuvsIndicesField.setAccessible(true);
      @SuppressWarnings("unchecked")
      var cuvsIndices = (IntObjectHashMap<CuVSIndex>) cuvsIndicesField.get(reader);

      // Find the field info for this field name
      var fieldInfosField = reader.getClass().getDeclaredField("fieldInfos");
      fieldInfosField.setAccessible(true);
      var fieldInfos = (FieldInfos) fieldInfosField.get(reader);

      FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);
      if (fieldInfo != null) {
        CuVSIndex cuvsIndex = cuvsIndices.get(fieldInfo.number);
        if (cuvsIndex != null) {
          return cuvsIndex.getCagraIndex();
        }
      }
    } catch (Exception e) {
      info("Failed to extract CAGRA index for field " + fieldName + ": " + e.getMessage());
    }
    return null;
  }

  /**
   * Writes a pre-built merged CAGRA index to the output.
   */
  private void writeMergedCagraIndex(FieldInfo fieldInfo, CagraIndex mergedIndex, int vectorCount)
      throws IOException {
    try {
      long cagraIndexOffset = cuvsIndex.getFilePointer();
      var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);

      // Serialize the merged index
      Path tmpFile = Files.createTempFile(resources.tempDirectory(), "mergedindex", "cag");
      mergedIndex.serialize(cagraIndexOutputStream, tmpFile);
      long cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;

      // Write metadata (assuming no brute force or HNSW indexes for merged result)
      writeMeta(fieldInfo, vectorCount, cagraIndexOffset, cagraIndexLength, 0L, 0L, 0L, 0L);

      // Clean up the merged index
      mergedIndex.destroyIndex();
    } catch (Throwable t) {
      handleThrowable(t);
    }
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    try {
      // Use CuVS CAGRA merge API
      if (indexType.cagra()) {
        mergeCagraIndexes(fieldInfo, mergeState);
      } else {
        // For non-CAGRA index types, use vector-based approach
        final FloatVectorValues mergedVectorValues =
            switch (fieldInfo.getVectorEncoding()) {
              case BYTE -> throw new AssertionError("bytes not supported");
              case FLOAT32 ->
                  KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
            };

        int vectorCount = mergedVectorValues.size();
        int dimensions = fieldInfo.getVectorDimension();

        CuVSMatrix dataset = getVectorDataMatrix(mergedVectorValues, vectorCount, dimensions);
        if (dataset == null) {
          writeEmpty(fieldInfo);
          return;
        }
        writeFieldInternal(fieldInfo, dataset);
      }
    } catch (Throwable t) {
      handleThrowable(t);
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorsWriter.finish();

    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (cuvsIndex != null) {
      CodecUtil.writeFooter(cuvsIndex);
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, cuvsIndex, flatVectorsWriter);
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}
