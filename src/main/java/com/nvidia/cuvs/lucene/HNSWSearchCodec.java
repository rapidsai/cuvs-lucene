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

import com.nvidia.cuvs.LibraryException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

/** CuVS based codec for GPU based vector search */
public class HNSWSearchCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(HNSWSearchCodec.class.getName());

  private static final int DEFAULT_CUVS_WRITER_THREADS = 1;
  private static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  private static final int DEFAULT_GRAPH_DEGREE = 64;
  private static final int DEFAULT_HNSW_LAYERS = 1;
  private static final String CLASS_NAME = "HNSWSearchCodec";

  private KnnVectorsFormat format;

  public HNSWSearchCodec() {
    this(CLASS_NAME, new Lucene101Codec());
  }

  public HNSWSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormatDefaultValues();
  }

  public HNSWSearchCodec(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree, int hnswLayers) {
    this(CLASS_NAME, new Lucene101Codec());
    initializeFormat(cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers);
  }

  private void initializeFormatDefaultValues() {
    initializeFormat(
        DEFAULT_CUVS_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_HNSW_LAYERS);
  }

  private void initializeFormat(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree, int hnswLayers) {
    try {
      format = new HNSWVectorsFormat(cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      log.severe("Couldn't load native library, possible classloader issue. " + ex.getMessage());
    }
  }

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return format;
  }

  public void setKnnFormat(KnnVectorsFormat format) {
    this.format = format;
  }
}
