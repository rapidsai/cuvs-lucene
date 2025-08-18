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

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/** CuVS based KnnVectorsFormat for GPU acceleration */
public class HNSWVectorsFormat extends KnnVectorsFormat {

  private static final Logger log = Logger.getLogger(HNSWVectorsFormat.class.getName());

  // TODO: fix Lucene version in name, to the final targeted release, if any
  static final String CUVS_META_CODEC_NAME = "Lucene102CuVSVectorsFormatMeta";
  static final String CUVS_META_CODEC_EXT = "vemc";
  static final String CUVS_INDEX_CODEC_NAME = "Lucene102CuVSVectorsFormatIndex";
  static final String CUVS_INDEX_EXT = "vcag";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;

  static final int DEFAULT_WRITER_THREADS = 32;
  static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  static final int DEFAULT_GRAPH_DEGREE = 64;
  static final int HNSW_GRAPH_LAYERS = 1;

  static CuVSResources resources = cuVSResourcesOrNull();

  /** The format for storing, reading, and merging raw vectors on disk. */
  private static final FlatVectorsFormat flatVectorsFormat =
      new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);

  final int maxDimensions = 4096;
  final int cuvsWriterThreads;
  final int intGraphDegree;
  final int graphDegree;
  final int hnswLayers; // Number of layers to create in CAGRA->HNSW conversion

  /**
   * Creates a CuVSVectorsFormat, with default values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public HNSWVectorsFormat() {
    this(
        DEFAULT_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        HNSW_GRAPH_LAYERS);
  }

  /**
   * Creates a CuVSVectorsFormat, with the given threads, graph degree, etc.
   *
   * @throws LibraryException if the native library fails to load
   */
  public HNSWVectorsFormat(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree, int hnswLayers) {
    super("CuVSVectorsFormat");
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.hnswLayers = hnswLayers;
  }

  private static CuVSResources cuVSResourcesOrNull() {
    try {
      resources = CuVSResources.create();
      return resources;
    } catch (UnsupportedOperationException uoe) {
      log.warning("cuvs is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      if (t instanceof ExceptionInInitializerError ex) {
        t = ex.getCause();
      }
      log.warning("Exception occurred during creation of cuvs resources. " + t);
    }
    return null;
  }

  /** Tells whether the platform supports cuvs. */
  public static boolean supported() {
    return resources != null;
  }

  private static void checkSupported() {
    if (!supported()) {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public HNSWVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    checkSupported();
    var flatWriter = flatVectorsFormat.fieldsWriter(state);
    return new HNSWVectorsWriter(
        state, cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, resources, flatWriter);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Lucene99HnswVectorsReader(state, flatVectorsFormat.fieldsReader(state));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return maxDimensions;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(this.getClass().getSimpleName());
    sb.append("(cuvsWriterThreads=").append(cuvsWriterThreads);
    sb.append("intGraphDegree=").append(intGraphDegree);
    sb.append("graphDegree=").append(graphDegree);
    sb.append("hnswLayers=").append(hnswLayers);
    sb.append("resources=").append(resources);
    sb.append(")");
    return sb.toString();
  }
}
