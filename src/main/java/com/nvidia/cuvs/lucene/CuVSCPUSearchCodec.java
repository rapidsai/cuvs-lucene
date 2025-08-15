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
import com.nvidia.cuvs.lucene.CuVSVectorsWriter.IndexType;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

/** CuVS based codec for GPU based vector search */
public class CuVSCPUSearchCodec extends FilterCodec {

  public CuVSCPUSearchCodec() {
    this("CuVSCPUSearchCodec", new Lucene101Codec());
  }

  public CuVSCPUSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormat();
  }

  public CuVSCPUSearchCodec(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree, int hnswLayers) {
    this("CuVSCPUSearchCodec", new Lucene101Codec());
    initializeFormat(cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers);
  }

  private void initializeFormat() {
    initializeFormat(1, 128, 64, 1); // Default values
  }

  private void initializeFormat(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree, int hnswLayers) {
    KnnVectorsFormat format;
    try {
      format =
          new CuVSVectorsFormat(
              cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, IndexType.HNSW_LUCENE);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      Logger log = Logger.getLogger(CuVSCodec.class.getName());
      log.severe("Couldn't load native library, possible classloader issue. " + ex.getMessage());
    }
  }

  KnnVectorsFormat knnFormat = null;

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return knnFormat;
  }

  public void setKnnFormat(KnnVectorsFormat format) {
    this.knnFormat = format;
  }
}
