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

import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.cuVSResourcesOrNull;
import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCagraToHnswSerializationAndSearch extends LuceneTestCase {

  protected static Logger log =
      Logger.getLogger(TestCagraToHnswSerializationAndSearch.class.getName());
  private static Random random;
  private static List<Path> indexDirPath;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", Lucene99AcceleratedHNSWVectorsFormat.supported());
    random = new Random();
    // Fixed seed so that we can validate against the same result.
    random.setSeed(222);
    indexDirPath = new ArrayList<Path>();
    for (int i = 0; i < 2; i++) {
      indexDirPath.add(Paths.get(UUID.randomUUID().toString()));
    }
  }

  @Test
  public void testCagraToHnswSerializationAndSearch() throws IOException {
    log.info("Test Scenario 1 - cuvs supported");
    test(
        false,
        new Integer[] {1869, 1803, 1302, 59, 1497, 108, 1411, 351, 1982},
        indexDirPath.get(0));

    log.info("Test Scenario 2 - cuvs NOT supported");
    test(true, new Integer[] {885, 612, 1795, 1806, 1665}, indexDirPath.get(1));
  }

  private void test(boolean disableResources, Integer[] expected, Path indexDirPath)
      throws IOException {

    Lucene101AcceleratedHNSWCodec codec =
        new Lucene101AcceleratedHNSWCodec(32, 128, 64, 3, 16, 100);

    if (disableResources) {
      Lucene99AcceleratedHNSWVectorsFormat.resources = null;
    }

    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);
    // TODO: handle random related issues.
    int numDocs = 2000; // random.nextInt(100, 1000);
    int dimension = 32; // random.nextInt(8, 1024);
    int topK = 5; // random.nextInt(5, 60);
    final int COMMIT_FREQ = 2000; // Math.min(numDocs, random.nextInt(100, 1000));
    int count = COMMIT_FREQ;
    final String ID_FIELD = "id";
    final String VECTOR_FIELD = "knn1";
    float[][] dataset = generateDataset(random, numDocs, dimension);

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        indexWriter.addDocument(document);
        count -= 1;
        if (count == 0) {
          indexWriter.commit();
          count = COMMIT_FREQ;
        }
      }
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath)) {
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
        log.info("Successfully opened index");

        int vectorCount = 0;
        for (LeafReaderContext leafReaderContext : reader.leaves()) {
          LeafReader leafReader = leafReaderContext.reader();
          FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD);
          assertNotNull(knnValues);
          log.info(
              VECTOR_FIELD
                  + " field: "
                  + knnValues.size()
                  + " vectors, "
                  + knnValues.dimension()
                  + " dimensions");
          vectorCount += knnValues.size();
          assertTrue("Vector dimension mismatch", knnValues.dimension() == dimension);
        }
        assertTrue("Dataset size mismatch", vectorCount == numDocs);

        log.info("Testing vector search queries...");
        IndexSearcher searcher = new IndexSearcher(reader);

        float[] queryVector = generateDataset(random, 1, dimension)[0];
        log.info("Query vector: " + Arrays.toString(queryVector));

        KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
        TopDocs results = searcher.search(query, topK);

        log.info("Search results (" + results.totalHits + " total hits):");
        HashSet<Integer> expectedIds = new HashSet<Integer>(Arrays.asList(expected));

        for (int i = 0; i < results.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          String id = doc.get(ID_FIELD);
          log.info(
              "  Rank "
                  + (i + 1)
                  + ": doc "
                  + scoreDoc.doc
                  + " (id="
                  + id
                  + "), score="
                  + scoreDoc.score);
          assertTrue(
              "Id " + id + " not found in expectedIds", expectedIds.contains(Integer.valueOf(id)));
        }
        assertTrue("TopK results not returned", results.scoreDocs.length == topK);

      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  @AfterClass
  public static void afterClass() throws Exception {
    // Reset resources.
    Lucene99AcceleratedHNSWVectorsFormat.resources = cuVSResourcesOrNull();
    // Cleanup.
    for (Path p : indexDirPath) {
      File indexDirPathFile = p.toFile();
      if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
        FileUtils.deleteDirectory(indexDirPathFile);
      }
    }
  }
}
