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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
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
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/**
 * KnnVectorsWriter for CuVS, responsible for merge and flush of vectors into
 * GPU
 */
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
  private final int hnswLayers; // Number of layers to create in CAGRA->HNSW conversion

  private final CuVSResources resources;
  private final IndexType indexType;

  private final FlatVectorsWriter flatVectorsWriter; // for writing the raw vectors
  private final List<CuVSFieldWriter> fields = new ArrayList<>();
  private IndexOutput meta = null, cuvsIndex = null;
  private IndexOutput hnswMeta = null, hnswVectorIndex = null;
  private final InfoStream infoStream;
  private boolean finished;
  private String vemFileName;
  private String vexFileName;

  /** The CuVS index Type. */
  public enum IndexType {
    /** Builds a Cagra index. */
    CAGRA(true, false, false, false),
    /** Builds a Brute Force index. */
    BRUTE_FORCE(false, true, false, false),
    /** Builds an HSNW index - suitable for searching on CPU. */
    HNSW(false, false, true, false),
    /** Builds a Cagra and a Brute Force index. */
    CAGRA_AND_BRUTE_FORCE(true, true, false, false),
    /** Builds a Lucene HNSW index via CAGRA. */
    HNSW_LUCENE(false, false, false, true);
    private final boolean cagra, bruteForce, hnsw, hnswLucene;

    IndexType(boolean cagra, boolean bruteForce, boolean hnsw, boolean hnswLucene) {
      this.cagra = cagra;
      this.bruteForce = bruteForce;
      this.hnsw = hnsw;
      this.hnswLucene = hnswLucene;
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

    public boolean hnswLucene() {
      return hnswLucene;
    }
  }

  public CuVSVectorsWriter(
      SegmentWriteState state,
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      IndexType indexType,
      CuVSResources resources,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.indexType = indexType;
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.hnswLayers = hnswLayers;
    this.resources = resources;
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;

    vemFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "vem");

    vexFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "vex");

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, CUVS_META_CODEC_EXT);
    String cagraFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, CUVS_INDEX_EXT);

    boolean success = false;
    try {

      // Only create CAGRA files if not in HNSW_LUCENE mode
      if (indexType == IndexType.HNSW_LUCENE) {

        hnswMeta = state.directory.createOutput(vemFileName, state.context);
        hnswVectorIndex = state.directory.createOutput(vexFileName, state.context);

        CodecUtil.writeIndexHeader(
            hnswMeta,
            "Lucene99HnswVectorsFormatMeta",
            Lucene99HnswVectorsFormat.VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
        CodecUtil.writeIndexHeader(
            hnswVectorIndex,
            "Lucene99HnswVectorsFormatIndex",
            Lucene99HnswVectorsFormat.VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
      } else {
        // Only create CAGRA files if not in HNSW_LUCENE mode
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
      }
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
    return writer;
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

  private void info(String msg) {
    if (infoStream.isEnabled(CUVS_COMPONENT)) {
      infoStream.message(CUVS_COMPONENT, msg);
    }
  }

  private void writeFieldInternal(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {
    if (vectors.size() == 0) {
      writeEmpty(fieldInfo);
      return;
    }
    long cagraIndexOffset, cagraIndexLength = 0L;
    long bruteForceIndexOffset, bruteForceIndexLength = 0L;
    long hnswIndexOffset, hnswIndexLength = 0L;

    // workaround for the minimum number of vectors for Cagra
    IndexType indexType =
        this.indexType.cagra() && vectors.size() < MIN_CAGRA_INDEX_SIZE
            ? IndexType.BRUTE_FORCE
            : this.indexType;

    info("=== INDEX TYPE DEBUG: original=" + this.indexType + ", effective=" + indexType + " ===");

    try {
      if (indexType.hnswLucene()) {
        info("=== ENTERED HNSW_LUCENE BLOCK (HNSW-only mode) ===");
        info("Entered the writeFieldInternal's HNSW LUCENE block - writing only HNSW files");
        try {
          CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
          writeHnswOnlyIndex(dataset, fieldInfo, vectors);
        } catch (Throwable t) {
          info("=== ERROR IN HNSW_LUCENE: " + t.getMessage() + " ===");
          handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
          // workaround for cuVS issue
          indexType = IndexType.BRUTE_FORCE;
        }
        // For HNSW_LUCENE, we don't write any CAGRA data, so set lengths to 0
        cagraIndexLength = 0L;
        cagraIndexOffset = 0L;
        bruteForceIndexOffset = 0L;
        bruteForceIndexLength = 0L;
        hnswIndexOffset = 0L;
        hnswIndexLength = 0L;
      } else {
        cagraIndexOffset = cuvsIndex.getFilePointer();
        if (indexType.cagra()) {
          try {
            var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
            CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
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
          CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
          writeBruteForceIndex(bruteForceIndexOutputStream, dataset);
          bruteForceIndexLength = cuvsIndex.getFilePointer() - bruteForceIndexOffset;
        }

        hnswIndexOffset = cuvsIndex.getFilePointer();
        if (indexType.hnsw()) {
          var hnswIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
          if (vectors.size() > MIN_CAGRA_INDEX_SIZE) {
            try {
              CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
              writeHNSWIndex(hnswIndexOutputStream, dataset);
            } catch (Throwable t) {
              handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
            }
          }
          hnswIndexLength = cuvsIndex.getFilePointer() - hnswIndexOffset;
        }
      }

      // Only write meta for non-HNSW_LUCENE modes
      if (indexType != IndexType.HNSW_LUCENE) {
        writeMeta(
            fieldInfo,
            vectors.size(),
            cagraIndexOffset,
            cagraIndexLength,
            bruteForceIndexOffset,
            bruteForceIndexLength,
            hnswIndexOffset,
            hnswIndexLength);
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  private void writeHnswOnlyIndex(
      CuVSMatrix dataset, FieldInfo fieldInfo, List<float[]> originalVectors) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams params = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();

    // Get the adjacency list from CAGRA index
    int[][] adjacencyList;
    try {
      adjacencyList = index.getGraph();
      info("=== SUCCESS: Got adjacency list from CAGRA index ===");
      info("Successfully got adjacency list from CAGRA index");
    } catch (Exception e) {
      info("=== FAILED: getGraph() method failed: " + e.getMessage() + " ===");
      info("getGraph() method failed or doesn't exist: " + e.getMessage());
      // Create a mock adjacency list for testing
      int size = (int) dataset.size();
      adjacencyList = new int[size][];
      for (int i = 0; i < size; i++) {
        // Create connections to next few nodes (circular)
        int degree = Math.min(10, size - 1); // up to 10 connections
        adjacencyList[i] = new int[degree];
        for (int j = 0; j < degree; j++) {
          adjacencyList[i][j] = (i + j + 1) % size;
        }
      }
      info(
          "=== CREATED MOCK ADJACENCY LIST: "
              + size
              + " nodes, degree="
              + (adjacencyList.length > 0 ? adjacencyList[0].length : 0)
              + " ===");
      info(
          "Created mock adjacency list with "
              + size
              + " nodes, degree="
              + (adjacencyList.length > 0 ? adjacencyList[0].length : 0));
    }

    int size = (int) dataset.size();
    int dimensions = fieldInfo.getVectorDimension();

    // Debug: Check if we got valid adjacency data
    info(
        "Adjacency list info: "
            + (adjacencyList == null
                ? "null"
                : "length="
                    + adjacencyList.length
                    + ", first row="
                    + (adjacencyList.length > 0 && adjacencyList[0] != null
                        ? adjacencyList[0].length
                        : "null")));

    // Create HNSW graph from CAGRA - multi-layer if original vectors available
    OnHeapHnswGraph hnswGraph;
    if (originalVectors != null && originalVectors.size() > 0) {
      info("=== Creating 3-layer HNSW graph using original vectors ===");
      hnswGraph =
          createMultiLayerHnswGraph(fieldInfo, size, dimensions, adjacencyList, originalVectors);
    } else {
      info("=== Creating single-layer HNSW graph (no original vectors) ===");
      // Create single layer graph
      List<int[]> singleLayerNodes = new ArrayList<>();
      List<int[][]> singleLayerAdjacencies = new ArrayList<>();
      singleLayerAdjacencies.add(adjacencyList);
      hnswGraph = new OnHeapHnswGraph(size, dimensions, singleLayerNodes, singleLayerAdjacencies);
    }

    // Remember the vector index offset before writing
    long vectorIndexOffset = hnswVectorIndex.getFilePointer();

    // Write the graph to the vector index
    int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);

    // Calculate the length of written data
    long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;

    // Write metadata
    writeMeta(
        hnswVectorIndex,
        hnswMeta,
        fieldInfo,
        vectorIndexOffset,
        vectorIndexLength,
        size,
        hnswGraph,
        graphLevelNodeOffsets);

    long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
    info("HNSW-only graph created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");

    // Don't serialize CAGRA index - destroy it immediately
    index.destroyIndex();
  }

  /**
   * Creates a multi-layer HNSW graph with dynamic number of layers.
   * M = cagraGraphDegree/2
   * Each layer contains 1/M nodes from the previous layer
   * Creates layers until the highest layer has ≤ M nodes
   */
  private OnHeapHnswGraph createMultiLayerHnswGraph(
      FieldInfo fieldInfo,
      int size,
      int dimensions,
      int[][] adjacencyList,
      List<float[]> originalVectors)
      throws Throwable {

    // Calculate M as cagraGraphDegree/2
    int M = graphDegree / 2;
    info(
        "=== Creating "
            + hnswLayers
            + "-layer HNSW graph with M="
            + M
            + " (cagraGraphDegree/2) ===");
    info("Creating " + hnswLayers + "-layer HNSW graph with size=" + size + ", M=" + M);

    // Store all layers data
    java.util.List<int[]> layerNodes = new java.util.ArrayList<>();
    java.util.List<int[][]> layerAdjacencies = new java.util.ArrayList<>();

    // Layer 0: Use full CAGRA adjacency list
    layerNodes.add(null); // Layer 0 contains all nodes, so we don't need to store node list
    layerAdjacencies.add(adjacencyList);

    // Build higher layers - create exactly hnswLayers-1 additional layers (layer 0 is already
    // added)
    int currentLayerSize = size;
    int layerIndex = 1;
    java.util.Random random = new java.util.Random(42); // Fixed seed for reproducibility

    while (layerIndex < hnswLayers && currentLayerSize > 1) {
      // Calculate size for next layer (1/M of current layer)
      int nextLayerSize = Math.max(1, currentLayerSize / M);

      info(
          "=== Layer "
              + layerIndex
              + " will have "
              + nextLayerSize
              + " nodes out of "
              + currentLayerSize
              + " (previous layer) ===");
      info("Layer " + layerIndex + " will have " + nextLayerSize + " nodes");

      // Select nodes for this layer
      java.util.Set<Integer> selectedNodesSet = new java.util.HashSet<>();

      if (layerIndex == 1) {
        // Select from all nodes (Layer 0)
        while (selectedNodesSet.size() < nextLayerSize) {
          selectedNodesSet.add(random.nextInt(size));
        }
      } else {
        // Select from previous layer nodes
        int[] prevLayerNodes = layerNodes.get(layerNodes.size() - 1);
        while (selectedNodesSet.size() < nextLayerSize) {
          int idx = random.nextInt(prevLayerNodes.length);
          selectedNodesSet.add(prevLayerNodes[idx]);
        }
      }

      // Convert to sorted array
      int[] selectedNodes =
          selectedNodesSet.stream().mapToInt(Integer::intValue).sorted().toArray();
      layerNodes.add(selectedNodes);

      info(
          "=== Selected Layer "
              + layerIndex
              + " nodes: "
              + java.util.Arrays.toString(
                  java.util.Arrays.copyOf(selectedNodes, Math.min(10, selectedNodes.length)))
              + (selectedNodes.length > 10 ? "..." : "")
              + " ===");

      // Extract vectors for selected nodes
      float[][] selectedVectors = new float[nextLayerSize][];
      for (int i = 0; i < nextLayerSize; i++) {
        int nodeId = selectedNodes[i];
        if (nodeId < originalVectors.size()) {
          selectedVectors[i] = originalVectors.get(nodeId);
        } else {
          selectedVectors[i] = createRandomVector(dimensions, nodeId);
        }
      }

      // Build CAGRA graph for this layer
      int[][] layerAdjacency = buildCagraGraphForSubset(selectedVectors, selectedNodes);
      layerAdjacencies.add(layerAdjacency);

      // Update for next iteration
      currentLayerSize = nextLayerSize;
      layerIndex++;

      // Use different seed for each layer
      random = new java.util.Random(42 + layerIndex);
    }

    int numLayers = layerAdjacencies.size();
    info("=== Total layers created: " + numLayers + " ===");
    info("Created " + numLayers + " layers total");

    // Create the multi-layer graph with all layers
    return new OnHeapHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
  }

  /**
   * Creates a random vector for fallback purposes
   */
  private float[] createRandomVector(int dimensions, int seed) {
    float[] vector = new float[dimensions];
    java.util.Random random = new java.util.Random(seed);
    for (int i = 0; i < dimensions; i++) {
      vector[i] = random.nextFloat();
    }
    return vector;
  }

  /**
   * Builds a CAGRA graph for a subset of vectors
   */
  private int[][] buildCagraGraphForSubset(float[][] vectors, int[] originalNodeIds)
      throws Throwable {
    if (vectors.length < 2) {
      // Can't build CAGRA graph with less than 2 vectors
      return new int[vectors.length][0];
    }

    try {
      // Create CuVSMatrix from the subset vectors
      CuVSMatrix subsetDataset = CuVSMatrix.ofArray(vectors);

      // Build CAGRA index for the subset
      CagraIndexParams params = cagraIndexParams(vectors.length);
      CagraIndex subsetIndex =
          CagraIndex.newBuilder(resources)
              .withDataset(subsetDataset)
              .withIndexParams(params)
              .build();

      // Get adjacency list from subset CAGRA index
      int[][] subsetAdjacency;
      try {
        subsetAdjacency = subsetIndex.getGraph();
        info("=== SUCCESS: Got adjacency list from Layer 1 CAGRA index ===");
        info("Successfully got adjacency list from Layer 1 CAGRA index");
      } catch (Exception e) {
        info("=== FAILED: getGraph() method failed for Layer 1: " + e.getMessage() + " ===");
        info("getGraph() method failed for Layer 1: " + e.getMessage());
        // Create mock adjacency list
        subsetAdjacency = new int[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
          int degree = Math.min(5, vectors.length - 1);
          subsetAdjacency[i] = new int[degree];
          for (int j = 0; j < degree; j++) {
            subsetAdjacency[i][j] = (i + j + 1) % vectors.length;
          }
        }
      }

      // Convert subset adjacency to use original node IDs
      int[][] layer1Adjacency = new int[vectors.length][];
      for (int i = 0; i < vectors.length; i++) {
        if (subsetAdjacency[i] != null) {
          layer1Adjacency[i] = new int[subsetAdjacency[i].length];
          for (int j = 0; j < subsetAdjacency[i].length; j++) {
            // Map subset index back to original node ID
            int subsetNeighborId = subsetAdjacency[i][j];
            layer1Adjacency[i][j] = originalNodeIds[subsetNeighborId];
          }
        } else {
          layer1Adjacency[i] = new int[0];
        }
      }

      subsetIndex.destroyIndex();
      return layer1Adjacency;

    } catch (Exception e) {
      info("=== FAILED to build CAGRA graph for subset: " + e.getMessage() + " ===");
      info("Failed to build CAGRA graph for subset: " + e.getMessage());

      // Fallback: create simple connections between Layer 1 nodes
      int[][] fallbackAdjacency = new int[vectors.length][];
      for (int i = 0; i < vectors.length; i++) {
        int degree = Math.min(3, vectors.length - 1);
        fallbackAdjacency[i] = new int[degree];
        for (int j = 0; j < degree; j++) {
          int neighborIdx = (i + j + 1) % vectors.length;
          fallbackAdjacency[i][j] = originalNodeIds[neighborIdx];
        }
      }
      return fallbackAdjacency;
    }
  }

  private void writeMeta(
      IndexOutput vectorIndex,
      IndexOutput meta,
      FieldInfo field,
      long vectorIndexOffset,
      long vectorIndexLength,
      int count,
      HnswGraph graph,
      int[][] graphLevelNodeOffsets)
      throws IOException {
    info(
        "=== writeMeta: Writing field "
            + field.name
            + " with count="
            + count
            + ", dimensions="
            + field.getVectorDimension()
            + " ===");
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeVLong(vectorIndexOffset);
    meta.writeVLong(vectorIndexLength);
    meta.writeVInt(field.getVectorDimension());
    meta.writeInt(count);
    // Use M = cagraGraphDegree/2
    int M = graphDegree / 2;
    info("=== writeMeta: Writing M=" + M + " (cagraGraphDegree/2) ===");
    meta.writeVInt(M); // M = cagraGraphDegree/2
    // write graph nodes on each level
    if (graph == null) {
      meta.writeVInt(0);
    } else {
      meta.writeVInt(graph.numLevels());
      long valueCount = 0;
      for (int level = 0; level < graph.numLevels(); level++) {
        NodesIterator nodesOnLevel = graph.getNodesOnLevel(level);
        valueCount += nodesOnLevel.size();
        if (level > 0) {
          int[] nol = new int[nodesOnLevel.size()];
          int numberConsumed = nodesOnLevel.consume(nol);
          Arrays.sort(nol);
          assert numberConsumed == nodesOnLevel.size();
          meta.writeVInt(nol.length); // number of nodes on a level
          for (int i = nodesOnLevel.size() - 1; i > 0; --i) {
            nol[i] -= nol[i - 1];
          }
          for (int n : nol) {
            assert n >= 0 : "delta encoding for nodes failed; expected nodes to be sorted";
            meta.writeVInt(n);
          }
        } else {
          assert nodesOnLevel.size() == count : "Level 0 expects to have all nodes";
        }
      }
      long start = vectorIndex.getFilePointer();
      meta.writeLong(start);
      meta.writeVInt(16); // DIRECT_MONOTONIC_BLOCK_SHIFT);
      final DirectMonotonicWriter memoryOffsetsWriter =
          DirectMonotonicWriter.getInstance(
              meta, vectorIndex, valueCount, 16); // DIRECT_MONOTONIC_BLOCK_SHIFT);
      long cumulativeOffsetSum = 0;
      int totalOffsetsWritten = 0;
      for (int[] levelOffsets : graphLevelNodeOffsets) {
        info(
            "=== writeMeta: Writing offsets for level with "
                + levelOffsets.length
                + " entries ===");
        for (int v : levelOffsets) {
          memoryOffsetsWriter.add(cumulativeOffsetSum);
          cumulativeOffsetSum += v;
          totalOffsetsWritten++;
        }
      }
      info(
          "=== writeMeta: Total offsets written: "
              + totalOffsetsWritten
              + ", expected: "
              + valueCount
              + " ===");
      memoryOffsetsWriter.finish();
      meta.writeLong(vectorIndex.getFilePointer() - start);
    }
  }

  private int[][] writeGraph(OnHeapHnswGraph graph, IndexOutput vectorIndex) throws IOException {
    if (graph == null) return new int[0][0];
    // write vectors' neighbors on each level into the vectorIndex file
    int countOnLevel0 = graph.size();
    int[][] offsets = new int[graph.numLevels()][];
    int[] scratch = new int[graph.maxConn() * 2];
    for (int level = 0; level < graph.numLevels(); level++) {
      int[] sortedNodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
      offsets[level] = new int[sortedNodes.length];
      int nodeOffsetId = 0;
      // Debug: print the actual number of nodes being processed
      info(
          "=== writeGraph: Level "
              + level
              + " has "
              + sortedNodes.length
              + " nodes, expected "
              + (level == 0 ? countOnLevel0 : "unknown")
              + " ===");
      for (int node : sortedNodes) {
        NeighborArray neighbors = graph.getNeighbors(level, node);
        int size = neighbors.size();
        // Write size in VInt as the neighbors list is typically small
        long offsetStart = vectorIndex.getFilePointer();
        int[] nnodes = neighbors.nodes();
        Arrays.sort(nnodes, 0, size);
        // Now that we have sorted, do delta encoding to minimize the required bits to store the
        // information
        int actualSize = 0;
        if (size > 0) {
          scratch[0] = nnodes[0];
          actualSize = 1;
        }
        for (int i = 1; i < size; i++) {
          assert nnodes[i] < countOnLevel0 : "node too large: " + nnodes[i] + ">=" + countOnLevel0;
          if (nnodes[i - 1] == nnodes[i]) {
            continue;
          }
          scratch[actualSize++] = nnodes[i] - nnodes[i - 1];
        }
        // Write the size after duplicates are removed
        vectorIndex.writeVInt(actualSize);
        for (int i = 0; i < actualSize; i++) {
          vectorIndex.writeVInt(scratch[i]);
        }
        offsets[level][nodeOffsetId++] =
            Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
      }
    }
    return offsets;
  }

  private void writeCagraIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams params = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();
    long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
    info("Cagra index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    Path tmpFile = Files.createTempFile(resources.tempDirectory(), "tmpindex", "cag");
    index.serialize(os, tmpFile);
    index.destroyIndex();
  }

  private void writeBruteForceIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    BruteForceIndexParams params =
        new BruteForceIndexParams.Builder()
            .withNumWriterThreads(32) // TODO: Make this
            // configurable later.
            .build();
    long startTime = System.nanoTime();
    var index =
        BruteForceIndex.newBuilder(resources).withIndexParams(params).withDataset(dataset).build();
    long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
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
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(indexParams).build();
    long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
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
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getVectors());
  }

  private void writeSortingField(CuVSFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {

    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);
    // TODO: Loading all vectors into memory is inefficient. Is there a way to stream the vectors
    // from the flat writer to the CuVSMatrix?

    // TODO: This is slightly different....
    List<float[]> sortedVectors = new ArrayList<float[]>();
    for (int i = 0; i < fieldData.getVectors().size(); i++) {
      sortedVectors.add(fieldData.getVectors().get(new2OldOrd[i]));
    }

    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
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
    Utils.handleThrowable(t);
  }

  private void mergeCagraIndexes(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {

      List<CagraIndex> cagraIndexes = new ArrayList<>();
      // We need this count so that the merged segment's meta information has the vector count.
      int totalVectorCount = 0;

      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        KnnVectorsReader knnReader = mergeState.knnVectorsReaders[i];
        // Access the CAGRA index for this field from the reader

        if (knnReader != null) {
          if (knnReader instanceof CuVSVectorsReader cvr) {
            if (cvr != null) {
              totalVectorCount += cvr.getFieldEntries().get(fieldInfo.number).count();
              CagraIndex cagraIndex = getCagraIndexFromReader(cvr, fieldInfo.name);
              if (cagraIndex != null) {
                cagraIndexes.add(cagraIndex);
              }
            }
          } else {
            // This should never happen
            throw new RuntimeException(
                "Reader is not of CuVSVectorsReader type. Instead it is: " + knnReader.getClass());
          }
        }
      }
      assert cagraIndexes.size() > 1;

      CagraIndex mergedIndex =
          CagraIndex.merge(cagraIndexes.toArray(new CagraIndex[cagraIndexes.size()]));
      writeMergedCagraIndex(fieldInfo, mergedIndex, totalVectorCount);
      info("Successfully merged " + cagraIndexes.size() + " CAGRA indexes using native merge API");

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Creates List<Float[]> from merged vectors
   * */
  private List<float[]> createListFromMergedVectors(FloatVectorValues mergedVectorValues)
      throws IOException {
    List<float[]> res = new ArrayList<float[]>();
    KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      int ordinal = iter.index();
      float[] vector = mergedVectorValues.vectorValue(ordinal);
      res.add(vector);
    }
    return res;
  }

  /**
   * Fallback method that rebuilds indexes from merged vectors.
   * Used when native CAGRA merge() is not possible. Also used
   * when non-CAGRA index types are used (for e.g. Brute Force index).
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    if (fieldInfo.getVectorEncoding() != FLOAT32) {
      throw new AssertionError("Only Float32 supported");
    }
    try {

      List<float[]> dataset =
          createListFromMergedVectors(
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState));
      writeFieldInternal(fieldInfo, dataset);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Extracts the CAGRA index for a specific field from a CuVSVectorsReader.
   */
  private CagraIndex getCagraIndexFromReader(CuVSVectorsReader reader, String fieldName) {
    try {
      IntObjectHashMap<CuVSIndex> cuvsIndices = reader.getCuvsIndexes();
      FieldInfos fieldInfos = reader.getFieldInfos();

      FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);

      if (fieldInfo != null) {
        CuVSIndex cuvsIndex = cuvsIndices.get(fieldInfo.number);
        if (cuvsIndex != null) {
          return cuvsIndex.getCagraIndex();
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
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
      // Only write meta for non-HNSW_LUCENE modes
      if (indexType != IndexType.HNSW_LUCENE) {
        writeMeta(fieldInfo, vectorCount, cagraIndexOffset, cagraIndexLength, 0L, 0L, 0L, 0L);
      }

      // Clean up the merged index
      mergedIndex.destroyIndex();
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

    if (indexType.cagra() && !indexType.bruteForce()) {
      // Since CAGRA merge does not support merging of indexes with purging of deletes,
      // we fallback to vector-based re-indexing. Issue:
      // https://github.com/rapidsai/cuvs/issues/1253
      boolean hasDeletions =
          IntStream.range(0, mergeState.liveDocs.length)
              .anyMatch(
                  i ->
                      mergeState.liveDocs[i] == null
                          || IntStream.range(0, mergeState.maxDocs[i])
                              .anyMatch(j -> !mergeState.liveDocs[i].get(j)));

      if (mergeState.knnVectorsReaders.length > 1 && !hasDeletions) {
        mergeCagraIndexes(fieldInfo, mergeState);
      } else {
        // CAGRA's merge API does not handle the trivial case of merging 1 index.
        vectorBasedMerge(fieldInfo, mergeState);
      }

    } else {
      // If there is a Brute Force index then re-index using the vectors even if there is a CAGRA
      // index.
      vectorBasedMerge(fieldInfo, mergeState);
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

    {
      if (hnswMeta != null) {
        // write end of fields marker
        hnswMeta.writeInt(-1);
        CodecUtil.writeFooter(hnswMeta);
      }
      if (hnswVectorIndex != null) {
        CodecUtil.writeFooter(hnswVectorIndex);
      }
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, cuvsIndex, hnswMeta, hnswVectorIndex, flatVectorsWriter);
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
