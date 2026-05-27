/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.util.Objects;

public class GPUSearchParams extends ParamsBase {

  public static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;

  private final IndexType indexType;

  /**
   * Constructs an instance of {@link GPUSearchParams} with specific parameter values.
   *
   * @param writerThreads Number of cuVS writer threads to use.
   * @param intermediateGraphDegree The intermediate graph degree while building the CAGRA index.
   * @param graphdegree The graph degree to use while building the CAGRA index.
   * @param cagraGraphBuildAlgo The CAGRA build algorithm to use.
   * @param indexType The type of index to build - CAGRA, BRUTEFORCE, or both.
   * @param cuVSIvfPqParams An instance of CuVSIvfPqParams containing IVF_PQ specific parameters.
   * @param strategy either HEURISTIC [Default] that automatically chooses build algorithm and its parameters based on data set size or CUSTOM that uses the parameters passed though this class.
   * @param heuristicType the heuristic type. The default option is SAME_GRAPH_FOOTPRINT.
   * @param cuvsDistanceType the cuvsDistanceType. The default option is L2Expanded.
   * @param nnDescentNumIterations the number of Iterations to run if building with NN_DESCENT.
   */
  private GPUSearchParams(
      int writerThreads,
      int intermediateGraphDegree,
      int graphdegree,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      IndexType indexType,
      CuVSIvfPqParams cuVSIvfPqParams,
      Strategy strategy,
      CuvsDistanceType cuvsDistanceType,
      int nnDescentNumIterations,
      int maxConn,
      int beamWidth) {
    super(
        writerThreads,
        intermediateGraphDegree,
        graphdegree,
        maxConn,
        beamWidth,
        cagraGraphBuildAlgo,
        cuVSIvfPqParams,
        strategy,
        cuvsDistanceType,
        nnDescentNumIterations);
    this.indexType = indexType;
  }

  /**
   * Get the index type parameter
   *
   * @return the index type parameter
   */
  public IndexType getIndexType() {
    return indexType;
  }

  @Override
  public String toString() {
    return super.toString() + " GPUSearchParams [indexType=" + indexType + "]";
  }

  /**
   * Builder class for creating an instance of {@link GPUSearchParams}
   */
  public static class Builder {

    private int writerThreads = DEFAULT_WRITER_THREADS;
    private int intermediateGraphDegree = DEFAULT_INT_GRAPH_DEGREE;
    private int graphdegree = DEFAULT_GRAPH_DEGREE;
    private CagraGraphBuildAlgo cagraGraphBuildAlgo = DEFAULT_CAGRA_GRAPH_BUILD_ALGO;
    private IndexType indexType = DEFAULT_INDEX_TYPE;
    private CuVSIvfPqParams cuVSIvfPqParams = null;
    private Strategy strategy = DEFAULT_STRATEGY;
    private CuvsDistanceType cuvsDistanceType = DEFAULT_CUVS_DISTANCE_TYPE;
    private int nnDescentNumIterations = DEFAULT_NN_DESCENT_NUM_ITERATIONS;
    private int maxConn = DEFAULT_MAX_CONN;
    private int beamWidth = DEFAULT_BEAM_WIDTH;

    /**
     * Set the number of cuVS writer threads while building the index
     * Valid range - Minimum: {@value MIN_WRITER_THREADS}, Maximum: {@value MAX_WRITER_THREADS}
     * Default value - {@value DEFAULT_WRITER_THREADS}
     *
     * @param writerThreads the number of cuVS writer threads
     * @return instance of {@link Builder}
     */
    public Builder withWriterThreads(int writerThreads) {
      this.writerThreads = writerThreads;
      return this;
    }

    /**
     * Set the intermediate graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_INT_GRAPH_DEG}, Maximum: {@value MAX_INT_GRAPH_DEG}
     * Default value - {@value DEFAULT_INT_GRAPH_DEGREE}
     *
     * @param intermediateGraphDegree the intermediate graph degree parameter
     * @return instance of {@link Builder}
     */
    public Builder withIntermediateGraphDegree(int intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    /**
     * Set the graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_GRAPH_DEG}, Maximum: {@value MAX_GRAPH_DEG}
     * Default value - {@value DEFAULT_GRAPH_DEGREE}
     *
     * @param graphDegree the graph degree parameter
     * @return instance of {@link Builder}
     */
    public Builder withGraphDegree(int graphDegree) {
      this.graphdegree = graphDegree;
      return this;
    }

    /**
     * Set the CAGRA build algorithm.
     * Cannot be null, defaults to NN_DESCENT
     *
     * @param cagraGraphBuildAlgo the CAGRA build algorithm to use
     * @return instance of {@link Builder}
     */
    public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cagraGraphBuildAlgo) {
      this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
      return this;
    }

    /**
     * Set the type of index to build - CAGRA, BRUTEFORCE, or both.
     * Cannot be null, defaults to CAGRA
     *
     * @param indexType the type of index to build
     * @return instance of {@link Builder}
     */
    public Builder withIndexType(IndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    /**
     * Set the instance of {@link CuVSIvfPqParams}
     *
     * @param cuVSIvfPqParams
     * @return instance of {@link Builder}
     */
    public Builder withCuVSIvfPqParams(CuVSIvfPqParams cuVSIvfPqParams) {
      this.cuVSIvfPqParams = cuVSIvfPqParams;
      return this;
    }

    /**
     * Set the chosen strategy:
     *
     * When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
     * When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally
     *
     * Valid options - HEURISTIC, CUSTOM
     * Default value - HEURISTIC
     *
     * @param strategy, the strategy to choose
     * @return instance of {@link Builder}
     */
    public Builder withStrategy(Strategy strategy) {
      this.strategy = strategy;
      return this;
    }

    /**
     * Set the CuvsDistanceType
     *
     * @param cuvsDistanceType the CuvsDistanceType to set
     * @return instance of {@link Builder}
     */
    public Builder withCuvsDistanceType(CuvsDistanceType cuvsDistanceType) {
      this.cuvsDistanceType = cuvsDistanceType;
      return this;
    }

    /**
     * Set the number of Iterations to run if building with NN_DESCENT
     *
     * Valid range - Minimum: {@value MIN_NN_DESCENT_NUM_ITERATIONS}, Maximum: {@value MAX_NN_DESCENT_NUM_ITERATIONS}
     * Default value - {@value DEFAULT_NN_DESCENT_NUM_ITERATIONS}
     *
     * @param nnDescentNumIterations number of merge workers to set
     * @return instance of {@link Builder}
     */
    public Builder withNNDescentNumIterations(int nnDescentNumIterations) {
      this.nnDescentNumIterations = nnDescentNumIterations;
      return this;
    }

    /**
     * Set the max connections parameter while building HNSW index with fallback mechanism
     * Valid range - Minimum: {@value MIN_MAX_CONN}, Maximum: {@value MAX_MAX_CONN}
     * Default value - {@value DEFAULT_MAX_CONN}
     *
     * @param maxConn the max connections parameter
     * @return instance of {@link Builder}
     */
    public Builder withMaxConn(int maxConn) {
      this.maxConn = maxConn;
      return this;
    }

    /**
     * Set the beam width parameter while building HNSW index with fallback mechanism
     * Valid range - Minimum: {@value MIN_BEAM_WIDTH}, Maximum: {@value MAX_BEAM_WIDTH}
     * Default value - {@value DEFAULT_BEAM_WIDTH}
     *
     * @param beamWidth the beam width parameter
     * @return instance of {@link Builder}
     */
    public Builder withBeamWidth(int beamWidth) {
      this.beamWidth = beamWidth;
      return this;
    }

    /**
     * Validates the input parameters.
     *
     * @throws IllegalArgumentException
     */
    private void validate() throws IllegalArgumentException {
      if (Objects.isNull(indexType)) {
        throw new IllegalArgumentException("indexType cannot be null.");
      }
    }

    /**
     * Creates and returns an instance of {@link GPUSearchParams}
     *
     * @return instance of {@link GPUSearchParams}
     */
    public GPUSearchParams build() {
      validate();
      return new GPUSearchParams(
          writerThreads,
          intermediateGraphDegree,
          graphdegree,
          cagraGraphBuildAlgo,
          indexType,
          cuVSIvfPqParams,
          strategy,
          cuvsDistanceType,
          nnDescentNumIterations,
          maxConn,
          beamWidth);
    }
  }
}
