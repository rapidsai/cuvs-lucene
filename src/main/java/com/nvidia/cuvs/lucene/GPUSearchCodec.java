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
import com.nvidia.cuvs.lucene.GPUVectorsWriter.IndexType;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

/** CuVS based codec for GPU based vector search */
public class GPUSearchCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(GPUSearchCodec.class.getName());
  private static final String NAME = "GPUSearchCodec";

  private static final int DEFAULT_CUVS_WRITER_THREADS = 1;
  private static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  private static final int DEFAULT_GRAPH_DEGREE = 64;
  private static final int DEFAULT_HNSW_LAYERS = 1;
  private static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;

  private KnnVectorsFormat format;

  public GPUSearchCodec() {
    this(NAME, new Lucene101Codec());
  }

  public GPUSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    try {
      format =
          new GPUVectorsFormat(
              DEFAULT_CUVS_WRITER_THREADS,
              DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
              DEFAULT_GRAPH_DEGREE,
              DEFAULT_HNSW_LAYERS,
              DEFAULT_INDEX_TYPE);
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
