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

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.codecs.Codec;
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
  private static Path indexDirPath;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", GPUVectorsFormat.supported());
    random = new Random();
    // Fixed seed so that we can validate against the same result.
    random.setSeed(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
  }

  @Test
  public void testCagraToHnswSerializationAndSearch() throws IOException {

    Codec codec = new HNSWSearchCodec();
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);

    int numDocs = 2000; // random.nextInt(100, 1000);
    int dimension = 32; // random.nextInt(8, 1024);
    int topK = 100; // random.nextInt(5, 60);
    final int COMMIT_FREQ = Math.min(numDocs, random.nextInt(100, 1000));
    int count = COMMIT_FREQ;
    final String VECTOR_FIELD = "knn1";
    float[][] dataset = generateDataset(random, numDocs, dimension);

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField("id", Integer.toString(i), Field.Store.YES));
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
          FloatVectorValues knnValues = leafReader.getFloatVectorValues("knn1");
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

        log.info("\n2. Testing vector search queries...");
        IndexSearcher searcher = new IndexSearcher(reader);

        float[] queryVector = generateDataset(random, 1, dimension)[0];
        log.info("Query vector: " + Arrays.toString(queryVector));

        KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
        TopDocs results = searcher.search(query, topK);

        log.info("\nknn1 search results (" + results.totalHits + " total hits):");
        int[] expected = {1803, 1869, 554, 1824, 1982, 1302, 320, 351, 707, 549};
        List<Integer> res = new ArrayList<Integer>();

        for (int i = 0; i < results.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          log.info(
              "  Rank "
                  + (i + 1)
                  + ": doc "
                  + scoreDoc.doc
                  + " (id="
                  + doc.get("id")
                  + "), score="
                  + scoreDoc.score);
          res.add(Integer.valueOf(doc.get("id")));
        }

        assertTrue("TopK results not returned", results.scoreDocs.length == topK);
        // TODO: make this test a bit more meaningful like checking the quality of search results.
        for (int i : expected) {
          assertTrue("Expected doc id is missing:" + i, res.contains(i));
        }

      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  @AfterClass
  public static void afterClass() throws Exception {
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }

  private static float[][] generateDataset(Random random, int datasetSize, int dimensions) {
    float[][] dataset = new float[datasetSize][dimensions];
    for (int i = 0; i < datasetSize; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }
}
