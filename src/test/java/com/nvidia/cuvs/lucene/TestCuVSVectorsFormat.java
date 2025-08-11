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

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.util.Arrays;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "Ignore the sysout")
public class TestCuVSVectorsFormat extends BaseKnnVectorsFormatTestCase {

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuvs is not supported", CuVSVectorsFormat.supported());
  }

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(new CuVSVectorsFormat());
    // For convenience, to sanitize the test code, one can comment out
    // the supported check and use another format, e.g.
    // return TestUtil.alwaysKnnVectorsFormat(new Lucene99HnswVectorsFormat());
  }

  @Test
  public void testMergeTwoSegsWithASingleDocPerSeg() throws Exception {
    float[][] f = new float[][] {randomVector(384), randomVector(384)};
    try (Directory dir = newDirectory();
        IndexWriter w =
            new IndexWriter(dir, newIndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      Document doc1 = new Document();
      doc1.add(new StringField("id", "0", Field.Store.NO));
      doc1.add(new KnnFloatVectorField("f", f[0], EUCLIDEAN));
      w.addDocument(doc1);
      w.commit();
      Document doc2 = new Document();
      doc2.add(new StringField("id", "1", Field.Store.NO));
      doc2.add(new KnnFloatVectorField("f", f[1], EUCLIDEAN));
      w.addDocument(doc2);
      w.flush();
      w.commit();

      // sanity - verify one doc per leaf
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        List<LeafReaderContext> subReaders = reader.leaves();
        assertEquals(2, subReaders.size());
        assertEquals(1, subReaders.get(0).reader().getFloatVectorValues("f").size());
        assertEquals(1, subReaders.get(1).reader().getFloatVectorValues("f").size());
      }

      // Enable merging for forceMerge to work
      w.getConfig().setMergePolicy(newMergePolicy());
      // now merge to a single segment
      w.forceMerge(1);

      // verify merged content
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues("f");
        assertNotNull(values);
        assertEquals(2, values.size());
        assertArrayEquals(f[0], values.vectorValue(0), 0.0f);
        assertArrayEquals(f[1], values.vectorValue(1), 0.0f);
      }
    }
  }

  // Basic test for multiple vectors fields per document
  @Test
  public void testTwoVectorFieldsPerDoc() throws Exception {
    float[][] f1 = new float[][] {randomVector(384), randomVector(384)};
    float[][] f2 = new float[][] {randomVector(384), randomVector(384)};
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      System.out.println("Is sorting enabled: " + w.getConfig().getIndexSort());
      Document doc1 = new Document();
      doc1.add(new StringField("id", "0", Field.Store.NO));
      doc1.add(new KnnFloatVectorField("f1", f1[0], EUCLIDEAN));
      doc1.add(new KnnFloatVectorField("f2", f2[0], EUCLIDEAN));
      w.addDocument(doc1);
      Document doc2 = new Document();
      doc2.add(new StringField("id", "1", Field.Store.NO));
      doc2.add(new KnnFloatVectorField("f1", f1[1], EUCLIDEAN));
      doc2.add(new KnnFloatVectorField("f2", f2[1], EUCLIDEAN));
      w.addDocument(doc2);

      System.out.println("Force merge - begin");
      w.forceMerge(1);
      System.out.println("Force merge - end");

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues("f1");
        assertNotNull(values);
        assertEquals(2, values.size());
        System.out.println("f1[0]" + Arrays.toString(f1[0]));
        System.out.println("f1[1]" + Arrays.toString(f1[1]));
        System.out.println("f2[0]" + Arrays.toString(f2[0]));
        System.out.println("f2[1]" + Arrays.toString(f2[1]));

        System.out.println("returned f1[0]" + Arrays.toString(values.vectorValue(0)));
        System.out.println("returned f1[1]" + Arrays.toString(values.vectorValue(1)));

        assertArrayEquals(f1[0], values.vectorValue(0), 0.0f);
        assertArrayEquals(f1[1], values.vectorValue(1), 0.0f);

        values = r.getFloatVectorValues("f2");
        assertNotNull(values);
        assertEquals(2, values.size());
        assertArrayEquals(f2[0], values.vectorValue(0), 0.0f);
        assertArrayEquals(f2[1], values.vectorValue(1), 0.0f);

        // opportunistically check boundary condition - search with a 0 topK
        var topDocs = r.searchNearestVectors("f1", randomVector(384), 0, null, 10);
        assertEquals(0, topDocs.scoreDocs.length);
        assertEquals(0, topDocs.totalHits.value());
      }
    }
  }

  // Override byte vector tests to skip them since we only support float32
  @Override
  public void testRandomBytes() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testSortedIndexBytes() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testMergingWithDifferentByteKnnFields() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testVectorValuesReportCorrectDocs() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testMismatchedFields() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testRandomExceptions() throws Exception {
    // Skip - this test uses byte vectors which are not supported
  }

  @Override
  public void testByteVectorScorerIteration() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testEmptyByteVectorData() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testCheckIntegrityReadsAllBytes() throws Exception {
    // Skip - may use byte vectors
  }

  @Override
  public void testSparseVectors() throws Exception {
    // Override to only test float vectors
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      int numDocs = atLeast(100);
      int dimension = atLeast(10);
      int numIndexed = 0;
      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        if (random().nextInt(4) == 3) {
          doc.add(new KnnFloatVectorField("knn", randomVector(dimension), EUCLIDEAN));
          numIndexed++;
        }
        doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues("knn");

        if (numIndexed == 0) {
          assertNull(values);
        } else {
          assertNotNull(values);
          assertEquals(numIndexed, values.size());
          assertEquals(dimension, values.dimension());
        }
      }
    }
  }
}
