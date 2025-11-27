/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_NAME;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;
import static org.apache.lucene.index.VectorEncoding.BYTE;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.logging.Logger;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/**
 * This class extends upon the KnnVectorsWriter to enable the creation of GPU-based accelerated
 * vector search indexes.
 *
 * @since 25.10
 */
public class Lucene99AcceleratedHNSWQuantizedVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(Lucene99AcceleratedHNSWQuantizedVectorsWriter.class);

  @SuppressWarnings("unused")
  private static final Logger log =
      Logger.getLogger(Lucene99AcceleratedHNSWQuantizedVectorsWriter.class.getName());

  /** The name of the CUVS component for the info-stream * */
  private static final String CUVS_COMPONENT = "CUVS";

  private final int cuvsWriterThreads;
  private final int intGraphDegree;
  private final int graphDegree;
  private final int hnswLayers; // Number of layers to create in CAGRA->HNSW conversion
  private final CuVSResources resources;
  private final FlatVectorsWriter flatVectorsWriter;
  private final List<ScalarQuantizedGPUFieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private IndexOutput cuvsIndex = null;
  private IndexOutput hnswMeta = null, hnswVectorIndex = null;
  private boolean finished;
  private String vemFileName;
  private String vexFileName;

  /**
   * Initializes {@link Lucene99AcceleratedHNSWQuantizedVectorsWriter}
   *
   * @param state instance of the {@link org.apache.lucene.index.SegmentWriteState}
   * @param cuvsWriterThreads number of cuVS threads to use while building the intermediate CAGRA index
   * @param intGraphDegree the intermediate graph degree to use while building the CAGRA index
   * @param graphDegree the graph degree to use while building the CAGRA index
   * @param hnswLayers the number of hnsw layers to construct while building the HNSW graph
   * @param resources instance of the {@link com.nvidia.cuvs.CuVSResources}
   * @param flatVectorsWriter instance of the {@link org.apache.lucene.codecs.hnsw.FlatVectorsWriter}
   * @throws IOException IOException
   */
  public Lucene99AcceleratedHNSWQuantizedVectorsWriter(
      SegmentWriteState state,
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      CuVSResources resources,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.hnswLayers = hnswLayers;
    this.resources = resources;
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;

    vemFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, HNSW_META_CODEC_EXT);

    vexFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, HNSW_INDEX_EXT);

    boolean success = false;
    try {

      hnswMeta = state.directory.createOutput(vemFileName, state.context);
      hnswVectorIndex = state.directory.createOutput(vexFileName, state.context);

      CodecUtil.writeIndexHeader(
          hnswMeta,
          HNSW_META_CODEC_NAME,
          Lucene99HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          hnswVectorIndex,
          HNSW_INDEX_CODEC_NAME,
          Lucene99HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);

      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Add new field for indexing.
   */
  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));

    if (encoding == FLOAT32) {
      @SuppressWarnings("unchecked")
      var flatWriter = (FlatFieldVectorsWriter<float[]>) writer;
      var cuvsFieldWriter = new ScalarQuantizedGPUFieldWriter(fieldInfo, flatWriter);
      fields.add(cuvsFieldWriter);
      return writer;
    } else if (encoding == BYTE) {
      @SuppressWarnings("unchecked")
      var flatWriter = (FlatFieldVectorsWriter<byte[]>) writer;
      var cuvsFieldWriter = new ScalarQuantizedGPUFieldWriter(fieldInfo, flatWriter);
      fields.add(cuvsFieldWriter);
      return writer;
    } else {
      throw new IllegalArgumentException(
          "expected FLOAT32 or BYTE encoding for scalar quantized vectors, got:" + encoding);
    }
  }

  /**
   * Utility method for building index metadata information string object.
   *
   * @param size index size
   * @param args additional metadata information
   * @return the string representation of the metadata information
   */
  static String indexMsg(int size, int... args) {
    StringBuilder sb = new StringBuilder("cagra index params");
    sb.append(": size=").append(size);
    sb.append(", intGraphDegree=").append(args[0]);
    sb.append(", actualIntGraphDegree=").append(args[1]);
    sb.append(", graphDegree=").append(args[2]);
    sb.append(", actualGraphDegree=").append(args[3]);
    return sb.toString();
  }

  /**
   * Builds an instance of CagraIndexParams.
   *
   * @return instance of CagraIndexParams
   */
  private CagraIndexParams cagraIndexParams() {
    return new CagraIndexParams.Builder()
        .withNumWriterThreads(cuvsWriterThreads)
        .withIntermediateGraphDegree(intGraphDegree)
        .withGraphDegree(graphDegree)
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .build();
  }

  /**
   * A utility method to print info/debugging messages using InfoStream.
   *
   * @param msg the debugging message to print
   */
  private void info(String msg) {
    if (infoStream.isEnabled(CUVS_COMPONENT)) {
      infoStream.message(CUVS_COMPONENT, msg);
    }
  }

  private static byte signedToUnsignedByte(byte signedByte) {
    return (byte) (signedByte & 0xFF);
  }

  private static byte[] convertSignedToUnsigned(byte[] signedVector) {
    byte[] unsignedVector = new byte[signedVector.length];
    for (int i = 0; i < signedVector.length; i++) {
      unsignedVector[i] = signedToUnsignedByte(signedVector[i]);
    }
    return unsignedVector;
  }

  /**
   * Builds the intermediate CAGRA index and builds and writes the HNSW index.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param vectors quantized vectors
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<byte[]> vectors) throws IOException {

    if (vectors.size() == 0) {
      writeEmpty(fieldInfo);
      return;
    }

    try {
      int dimensions = fieldInfo.getVectorDimension();

      // Convert 7-bit signed bytes to 8-bit unsigned bytes for cuVS compatibility
      List<byte[]> unsignedVectors = new ArrayList<>(vectors.size());
      for (byte[] signedVector : vectors) {
        unsignedVectors.add(convertSignedToUnsigned(signedVector));
      }

      // Create CuVSMatrix with BYTE data type (unsigned bytes)
      CuVSMatrix dataset = Utils.createByteMatrix(unsignedVectors, dimensions, resources);

      if (dataset.size() < 2) {
        writeSingleVectorGraph(fieldInfo, unsignedVectors);
        return;
      }

      long startTime = System.nanoTime();
      CagraIndexParams params = cagraIndexParams();
      CagraIndex cagraIndex =
          CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();

      CuVSMatrix adjacencyListMatrix = cagraIndex.getGraph();

      int size = (int) dataset.size();
      GPUBuiltHnswGraph hnswGraph =
          createMultiLayerHnswGraph(
              fieldInfo, size, dimensions, adjacencyListMatrix, unsignedVectors, hnswLayers);

      long vectorIndexOffset = hnswVectorIndex.getFilePointer();

      // Write the graph to the vector index
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);

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
      info(
          "HNSW graph created in "
              + elapsedMillis
              + "ms, with "
              + dataset.size()
              + " scalar quantized vectors");

      cagraIndex.close();

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Creates a multi-layer HNSW graph with dynamic number of layers.
   * M = cagraGraphDegree/2
   * Each layer contains 1/M nodes from the previous layer
   * Creates layers until the highest layer has ≤ M nodes
   */
  private GPUBuiltHnswGraph createMultiLayerHnswGraph(
      FieldInfo fieldInfo,
      int size,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      List<byte[]> vectors,
      int hnswLayers)
      throws Throwable {

    // Calculate M as cagraGraphDegree/2
    int M = graphDegree / 2;

    // Store all layers data
    List<int[]> layerNodes = new ArrayList<>();
    List<CuVSMatrix> layerAdjacencies = new ArrayList<>();

    // Layer 0: Use full CAGRA adjacency list
    layerNodes.add(null); // Layer 0 contains all nodes, so we don't need to store node list
    layerAdjacencies.add(adjacencyListMatrix);

    int currentLayerSize = size;
    int layerIndex = 1;
    Random random = new Random();

    while (layerIndex < hnswLayers && currentLayerSize > 1) {
      // Calculate size for next layer (1/M of current layer)
      int nextLayerSize = Math.max(2, currentLayerSize / M);
      // Select nodes for this layer
      SortedSet<Integer> selectedNodesSet = new TreeSet<>();

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

      // Extract vectors for selected nodes
      byte[][] selectedVectors = new byte[nextLayerSize][];
      for (int i = 0; i < nextLayerSize; i++) {
        selectedVectors[i] = vectors.get(selectedNodes[i]);
      }

      // Build CAGRA graph for this layer
      layerAdjacencies.add(buildCagraGraphForSubset(selectedVectors, selectedNodes, dimensions));

      // Update for next iteration
      currentLayerSize = nextLayerSize;
      layerIndex++;

      // Use different seed for each layer
      random = new Random(new Random().nextLong());
    }

    // Create the multi-layer graph with all layers
    return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
  }

  /**
   * Builds a CAGRA graph for a subset of scalar quantized vectors
   */
  private CuVSMatrix buildCagraGraphForSubset(byte[][] vectors, int[] selectedNodes, int dimensions)
      throws Throwable {
    // Create CuVSMatrix from the subset vectors (already converted to unsigned)
    CuVSMatrix subsetDataset = Utils.createByteMatrixFromArray(vectors, dimensions, resources);

    // Build CAGRA index for the subset
    CagraIndexParams params = cagraIndexParams();
    CagraIndex subsetIndex =
        CagraIndex.newBuilder(resources).withDataset(subsetDataset).withIndexParams(params).build();

    // Get adjacency list from subset CAGRA index
    CuVSMatrix cagraGraph = subsetIndex.getGraph();

    long numNodes = cagraGraph.size();
    long degree = cagraGraph.columns();

    // Create a re-mapped adjacency list
    int[][] remappedAdjacency = new int[(int) numNodes][(int) degree];

    for (int i = 0; i < numNodes; i++) {
      RowView rv = cagraGraph.getRow(i);
      for (int j = 0; j < degree && j < rv.size(); j++) {
        int subsetIndex1 = rv.getAsInt(j);
        // Map subset index to original node ID
        if (subsetIndex1 >= 0 && subsetIndex1 < selectedNodes.length) {
          remappedAdjacency[i][j] = selectedNodes[subsetIndex1];
        } else {
          // Invalid index, use self-reference
          remappedAdjacency[i][j] = selectedNodes[i];
        }
      }
    }

    subsetIndex.close();
    return CuVSMatrix.ofArray(remappedAdjacency);
  }

  /**
   * Writes the meta information for the index.
   *
   * @param vectorIndex instance of IndexOutput
   * @param meta instance of IndexOutput
   * @param field instance of FieldInfo
   * @param vectorIndexOffset vector index offset
   * @param vectorIndexLength vector index length
   * @param count the count of vectors
   * @param graph instance of HnswGraph
   * @param graphLevelNodeOffsets graph level node offsets
   * @throws IOException I/O Exceptions
   */
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

    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeVLong(vectorIndexOffset);
    meta.writeVLong(vectorIndexLength);
    meta.writeVInt(field.getVectorDimension());
    meta.writeInt(count);
    meta.writeVInt(graphDegree / 2); // M = cagraGraphDegree/2

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
          DirectMonotonicWriter.getInstance(meta, vectorIndex, valueCount, 16);
      long cumulativeOffsetSum = 0;
      for (int[] levelOffsets : graphLevelNodeOffsets) {
        for (int v : levelOffsets) {
          memoryOffsetsWriter.add(cumulativeOffsetSum);
          cumulativeOffsetSum += v;
        }
      }

      memoryOffsetsWriter.finish();

      meta.writeLong(vectorIndex.getFilePointer() - start);
    }
  }

  /**
   * Returns a 2D array of offsets (information written while writing the meta info)
   *
   * @param graph instance of GPUBuiltHnswGraph
   * @param vectorIndex instance of IndexOutput
   * @return a 2D array of offsets
   * @throws IOException I/O Exceptions
   */
  private int[][] writeGraph(GPUBuiltHnswGraph graph, IndexOutput vectorIndex) throws IOException {
    // write vectors' neighbors on each level into the vectorIndex file
    int countOnLevel0 = graph.size();
    int[][] offsets = new int[graph.numLevels()][];
    int[] scratch = new int[graph.maxConn() * 2];
    for (int level = 0; level < graph.numLevels(); level++) {
      int[] sortedNodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
      offsets[level] = new int[sortedNodes.length];
      int nodeOffsetId = 0;

      for (int node : sortedNodes) {
        // Get node neighbors
        NeighborArray neighbors = graph.getNeighbors(level, node);
        // Get the size of the neighbor array
        int size = neighbors.size();
        // Write size in VInt as the neighbors list is typically small
        long offsetStart = vectorIndex.getFilePointer();
        // Get neighbors
        int[] nnodes = neighbors.nodes();
        // Sort them
        Arrays.sort(nnodes, 0, size);
        // Now that we have sorted, do delta encoding to minimize the required bits to store the
        // information
        int actualSize = 0;
        if (size > 0) {
          scratch[0] = nnodes[0];
          actualSize = 1;
        }
        // De-duplication
        for (int i = 1; i < size; i++) {
          assert nnodes[i] < countOnLevel0 : "node too large: " + nnodes[i] + ">=" + countOnLevel0;
          // Sorting step helps here
          if (nnodes[i - 1] == nnodes[i]) {
            continue;
          }
          scratch[actualSize++] = nnodes[i] - nnodes[i - 1];
        }
        // Write the size after duplicates are removed
        vectorIndex.writeVInt(actualSize);
        // Write de-duplicated neighbors
        for (int i = 0; i < actualSize; i++) {
          vectorIndex.writeVInt(scratch[i]);
        }
        offsets[level][nodeOffsetId++] =
            Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
      }
    }
    // Return offsets (information written while writing the meta info)
    return offsets;
  }

  /**
   * Build the indexes and writes it to the disk.
   */
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

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData
   * @throws IOException
   */
  private void writeField(ScalarQuantizedGPUFieldWriter fieldData) throws IOException {
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getVectors());
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData instance of ScalarQuantizedGPUFieldWriter
   * @param sortMap instance of the DocMap
   * @throws IOException
   */
  private void writeSortingField(ScalarQuantizedGPUFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {

    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);

    List<byte[]> sortedVectors = new ArrayList<byte[]>();
    for (int i = 0; i < fieldData.getVectors().size(); i++) {
      sortedVectors.add(fieldData.getVectors().get(new2OldOrd[i]));
    }

    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
  }

  /**
   * Builds and writes a single vector graph.
   *
   * @param fieldInfo instance of FieldInfo
   * @param vectors the list of scalar quantized vectors (already converted to unsigned)
   * @throws IOException I/O Exceptions
   */
  private void writeSingleVectorGraph(FieldInfo fieldInfo, List<byte[]> vectors)
      throws IOException {
    // Workaround for CAGRA not supporting single vector indexes
    try {
      int size = 1;
      int dimensions = fieldInfo.getVectorDimension();

      // Create a dummy HNSW graph for a single vector
      GPUBuiltHnswGraph hnswGraph = createSingleVectorHnswGraph(size, dimensions);

      long vectorIndexOffset = hnswVectorIndex.getFilePointer();

      // Write the graph to the vector index
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);

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

      long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - System.nanoTime());
      info("Single vector HNSW graph created in " + elapsedMillis + "ms, with " + size + " vector");

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Creates a dummy HNSW graph for a single vector.
   * The graph will have 1 level with 1 node and no neighbors.
   */
  private GPUBuiltHnswGraph createSingleVectorHnswGraph(int size, int dimensions) throws Throwable {
    // Create adjacency list for single node with no neighbors
    int[][] singleNodeAdjacency = new int[][] {{-1}}; // -1 indicates no neighbors

    // Create CuVSMatrix from the adjacency list
    CuVSMatrix adjacencyMatrix = CuVSMatrix.ofArray(singleNodeAdjacency);

    // Create layer data for single-level graph
    List<int[]> layerNodes = new ArrayList<>();
    List<CuVSMatrix> layerAdjacencies = new ArrayList<>();

    // Layer 0: contains all nodes (just the single node)
    layerNodes.add(null); // Layer 0 contains all nodes, so we don't need to store node list
    layerAdjacencies.add(adjacencyMatrix);

    // Create the single-layer graph
    return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
  }

  /**
   * Writes an empty meta information for the field.
   *
   * @param fieldInfo instance of FieldInfo
   * @throws IOException I/O Exceptions
   */
  private void writeEmpty(FieldInfo fieldInfo) throws IOException {
    writeMeta(null, hnswMeta, fieldInfo, 0, 0, 0, null, null);
  }

  static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < SIMILARITY_FUNCTIONS.size(); i++) {
      if (SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  /**
   * Write field for merging.
   */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

    // Rebuild HNSW index from merged quantized vectors
    // Similar to Lucene99AcceleratedHNSWVectorsWriter.vectorBasedMerge
    vectorBasedMerge(fieldInfo, mergeState);
  }

  /**
   * Fallback method that rebuilds indexes from merged vectors.
   * Used when native CAGRA merge() is not possible.
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      FloatVectorValues mergedVectorValues =
          KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);

      if (mergedVectorValues != null) {
        List<float[]> floatVectors = new ArrayList<>();
        KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
        for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
          float[] vector = mergedVectorValues.vectorValue(iter.index());
          floatVectors.add(vector);
        }

        if (!floatVectors.isEmpty()) {
          int dimensions = floatVectors.get(0).length;
          int numVectors = floatVectors.size();

          float[] minPerDim = new float[dimensions];
          float[] maxPerDim = new float[dimensions];
          Arrays.fill(minPerDim, Float.MAX_VALUE);
          Arrays.fill(maxPerDim, Float.MIN_VALUE);

          for (float[] vector : floatVectors) {
            for (int d = 0; d < dimensions; d++) {
              minPerDim[d] = Math.min(minPerDim[d], vector[d]);
              maxPerDim[d] = Math.max(maxPerDim[d], vector[d]);
            }
          }

          List<byte[]> dataset = new ArrayList<>(numVectors);
          for (float[] vector : floatVectors) {
            byte[] quantized = new byte[dimensions];
            for (int d = 0; d < dimensions; d++) {
              float value = vector[d];
              float min = minPerDim[d];
              float max = maxPerDim[d];

              byte signedByte;
              if (max - min == 0) {
                signedByte = 0;
              } else {
                float normalized = (value - min) / (max - min);
                signedByte = (byte) Math.round(normalized * 127 - 64);
              }

              quantized[d] = signedToUnsignedByte(signedByte);
            }
            dataset.add(quantized);
          }
          writeFieldInternal(fieldInfo, dataset);
        }
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Called once at the end before close.
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorsWriter.finish();

    if (cuvsIndex != null) {
      CodecUtil.writeFooter(cuvsIndex);
    }

    if (hnswMeta != null) {
      // write end of fields marker
      hnswMeta.writeInt(-1);
      CodecUtil.writeFooter(hnswMeta);
    }
    if (hnswVectorIndex != null) {
      CodecUtil.writeFooter(hnswVectorIndex);
    }
  }

  /**
   * Closes the resources.
   */
  @Override
  public void close() throws IOException {
    IOUtils.close(cuvsIndex, hnswMeta, hnswVectorIndex, flatVectorsWriter);
  }

  /**
   * Returns the memory usage of this object in bytes.
   */
  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }

  /**
   * Helper class for scalar quantized field writer
   */
  /**
   * Helper class for scalar quantized field writer
   * Handles both FLOAT32 (quantizes to 7-bit signed bytes) and BYTE (already quantized) encodings
   */
  private static class ScalarQuantizedGPUFieldWriter extends KnnFieldVectorsWriter<Object> {

    private static final long SHALLOW_SIZE =
        org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance(
            ScalarQuantizedGPUFieldWriter.class);

    private final FieldInfo fieldInfo;
    private final FlatFieldVectorsWriter<?> flatFieldVectorsWriter;
    private final boolean isFloatEncoding;
    private int lastDocID = -1;

    @SuppressWarnings("unchecked")
    public ScalarQuantizedGPUFieldWriter(
        FieldInfo fieldInfo, FlatFieldVectorsWriter<?> flatFieldVectorsWriter) {
      this.fieldInfo = fieldInfo;
      this.flatFieldVectorsWriter = flatFieldVectorsWriter;
      this.isFloatEncoding = fieldInfo.getVectorEncoding() == FLOAT32;
    }

    @Override
    public void addValue(int docID, Object vectorValue) throws IOException {
      if (docID == lastDocID) {
        throw new IllegalArgumentException(
            "VectorValuesField \""
                + fieldInfo.name
                + "\" appears more than once in this document (only one value is allowed per"
                + " field)");
      }
      if (isFloatEncoding) {
        @SuppressWarnings("unchecked")
        FlatFieldVectorsWriter<float[]> floatWriter =
            (FlatFieldVectorsWriter<float[]>) flatFieldVectorsWriter;
        floatWriter.addValue(docID, (float[]) vectorValue);
      } else {
        @SuppressWarnings("unchecked")
        FlatFieldVectorsWriter<byte[]> byteWriter =
            (FlatFieldVectorsWriter<byte[]>) flatFieldVectorsWriter;
        byteWriter.addValue(docID, (byte[]) vectorValue);
      }
    }

    List<byte[]> getVectors() {
      if (isFloatEncoding) {
        @SuppressWarnings("unchecked")
        FlatFieldVectorsWriter<float[]> floatWriter =
            (FlatFieldVectorsWriter<float[]>) flatFieldVectorsWriter;
        List<float[]> floatVectors = floatWriter.getVectors();
        return quantizeFloatVectors(floatVectors);
      } else {
        @SuppressWarnings("unchecked")
        FlatFieldVectorsWriter<byte[]> byteWriter =
            (FlatFieldVectorsWriter<byte[]>) flatFieldVectorsWriter;
        return byteWriter.getVectors();
      }
    }

    private List<byte[]> quantizeFloatVectors(List<float[]> floatVectors) {
      if (floatVectors.isEmpty()) {
        return new ArrayList<>();
      }

      int dimensions = floatVectors.get(0).length;
      int numVectors = floatVectors.size();

      float[] minPerDim = new float[dimensions];
      float[] maxPerDim = new float[dimensions];
      Arrays.fill(minPerDim, Float.MAX_VALUE);
      Arrays.fill(maxPerDim, Float.MIN_VALUE);

      for (float[] vector : floatVectors) {
        for (int d = 0; d < dimensions; d++) {
          minPerDim[d] = Math.min(minPerDim[d], vector[d]);
          maxPerDim[d] = Math.max(maxPerDim[d], vector[d]);
        }
      }

      // Quantize to 7-bit signed bytes (-64 to 63)
      List<byte[]> quantizedVectors = new ArrayList<>(numVectors);
      for (float[] vector : floatVectors) {
        byte[] quantized = new byte[dimensions];
        for (int d = 0; d < dimensions; d++) {
          float range = maxPerDim[d] - minPerDim[d];
          if (range > 0) {
            float normalized = (vector[d] - minPerDim[d]) / range;
            int quantizedValue = Math.round(normalized * 127.0f) - 64;
            quantized[d] = (byte) Math.max(-64, Math.min(63, quantizedValue));
          } else {
            quantized[d] = 0;
          }
        }
        quantizedVectors.add(quantized);
      }

      return quantizedVectors;
    }

    FieldInfo fieldInfo() {
      return fieldInfo;
    }

    DocsWithFieldSet getDocsWithFieldSet() {
      return flatFieldVectorsWriter.getDocsWithFieldSet();
    }

    @Override
    public Object copyValue(Object vectorValue) {
      throw new UnsupportedOperationException();
    }

    @Override
    public long ramBytesUsed() {
      return SHALLOW_SIZE + flatFieldVectorsWriter.ramBytesUsed();
    }
  }
}
