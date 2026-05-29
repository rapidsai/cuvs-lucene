/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSIvfPqParams;
import java.util.Objects;
import java.util.function.Supplier;

public class ParamsBase {

  public static enum Strategy {
    /*
     * This strategy allows for automatic selection of the underlying CAGRA build algorithm.
     * With this strategy we use NN_DESCENT for dataset less than 5M vectors, else we use IVF_PQ.
     * Indexing parameters, especially for IVF_PQ, are heuristically identified automatically.
     *
     * This is the default and the recommended strategy.
     */
    HEURISTIC,
    /*
     * This is an option when the end-user would want to use custom parameter values.
     * This strategy should only be used under expert guidance.
     */
    CUSTOM
  }

  /*
   * TODO: Update boundaries for all parameters when a consensus is reached.
   * Issue: https://github.com/rapidsai/cuvs-lucene/issues/99
   */
  public static final int MIN_WRITER_THREADS = 1;
  public static final int MAX_WRITER_THREADS = 512;
  public static final int MIN_INT_GRAPH_DEG = 2;
  public static final int MAX_INT_GRAPH_DEG = 512;
  public static final int MIN_GRAPH_DEG = 1;
  public static final int MAX_GRAPH_DEG = 512;
  public static final int MIN_MAX_CONN = 1;
  public static final int MAX_MAX_CONN = 512;
  public static final int MIN_BEAM_WIDTH = 1;
  public static final int MAX_BEAM_WIDTH = 512;
  public static final int MIN_NN_DESCENT_NUM_ITERATIONS = 1;
  public static final int MAX_NN_DESCENT_NUM_ITERATIONS = 100;

  public static final int DEFAULT_WRITER_THREADS = 1;
  public static final int DEFAULT_INT_GRAPH_DEGREE = 128;
  public static final int DEFAULT_GRAPH_DEGREE = 64;
  public static final int DEFAULT_MAX_CONN = 32;
  public static final int DEFAULT_BEAM_WIDTH = 32;
  public static final CagraGraphBuildAlgo DEFAULT_CAGRA_GRAPH_BUILD_ALGO =
      CagraGraphBuildAlgo.NN_DESCENT;
  public static final Strategy DEFAULT_STRATEGY = Strategy.HEURISTIC;
  public static final CuvsDistanceType DEFAULT_CUVS_DISTANCE_TYPE = CuvsDistanceType.L2Expanded;
  public static final int DEFAULT_NN_DESCENT_NUM_ITERATIONS = 20;

  public static final Supplier<CuVSIvfPqParams> DEFAULT_IVF_PQ_PARAMS =
      () -> {
        return new CuVSIvfPqParams.Builder().build();
      };

  private final int writerThreads;
  private final int intermediateGraphDegree;
  private final int graphdegree;
  private final int maxConn;
  private final int beamWidth;
  private final CagraGraphBuildAlgo cagraGraphBuildAlgo;
  private final CuVSIvfPqParams cuVSIvfPqParams;
  private final Strategy strategy;
  private final CuvsDistanceType cuvsDistanceType;
  private final int nnDescentNumIterations;

  /**
   * Constructs an instance of {@link AcceleratedHNSWParams} with specific parameter values.
   *
   * @param writerThreads Number of cuVS writer threads to use.
   * @param intermediateGraphDegree The intermediate graph degree while building the CAGRA index.
   * @param graphdegree The graph degree to use while building the CAGRA index.
   * @param maxConn The max connection parameter used when building HNSW index with the fallback mechanism.
   * @param beamWidth The beam width parameter used when building HNSW index with the fallback mechanism.
   * @param cagraGraphBuildAlgo The CAGRA graph build algorithm to use [NN_DESCENT, IVF_PQ].
   * @param cuVSIvfPqParams An instance of CuVSIvfPqParams containing IVF_PQ specific parameters.
   * @param strategy either HEURISTIC [Default] that automatically chooses build algorithm and its parameters based on data set size or CUSTOM that uses the parameters passed though this class.
   * @param cuvsDistanceType the cuvsDistanceType. The default option is L2Expanded.
   * @param nnDescentNumIterations the number of Iterations to run if building with NN_DESCENT.
   */
  protected ParamsBase(
      int writerThreads,
      int intermediateGraphDegree,
      int graphdegree,
      int maxConn,
      int beamWidth,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      CuVSIvfPqParams cuVSIvfPqParams,
      Strategy strategy,
      CuvsDistanceType cuvsDistanceType,
      int nnDescentNumIterations) {
    this.writerThreads = writerThreads;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphdegree = graphdegree;
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
    this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
    this.cuVSIvfPqParams = cuVSIvfPqParams;
    this.strategy = strategy;
    this.cuvsDistanceType = cuvsDistanceType;
    this.nnDescentNumIterations = nnDescentNumIterations;
    validate();
    if (Objects.isNull(cuVSIvfPqParams)) {
      cuVSIvfPqParams = DEFAULT_IVF_PQ_PARAMS.get();
    }
  }

  /**
   * Get the cuVS writer threads parameter
   *
   * @return cuVS writer threads parameter
   */
  public int getWriterThreads() {
    return writerThreads;
  }

  /**
   * Get the intermediate graph degree
   *
   * @return the graph degree parameter
   */
  public int getIntermediateGraphDegree() {
    return intermediateGraphDegree;
  }

  /**
   * Get the graph degree
   *
   * @return the graph degree parameter
   */
  public int getGraphdegree() {
    return graphdegree;
  }

  /**
   * Get the max connection parameter
   *
   * @return the max connection parameter
   */
  public int getMaxConn() {
    return maxConn;
  }

  /**
   * Get the beam width parameter
   *
   * @return the beam width parameter
   */
  public int getBeamWidth() {
    return beamWidth;
  }

  /**
   * Get the CAGRA graph build algorithm
   *
   * @return the CAGRA graph build algorithm
   */
  public CagraGraphBuildAlgo getCagraGraphBuildAlgo() {
    return cagraGraphBuildAlgo;
  }

  /**
   * Get the instance of {@link CuVSIvfPqParams}
   *private
   * @return the instance of {@link CuVSIvfPqParams}
   */
  public CuVSIvfPqParams getCuVSIvfPqParams() {
    return cuVSIvfPqParams;
  }

  /**
   * Get the chosen strategy:
   *
   * When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
   * When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally
   *
   * @return get the chosen {@link Strategy}
   */
  public Strategy getStrategy() {
    return strategy;
  }

  /**
   * Get the cuvs distance type
   *
   * @return the distance type
   */
  public CuvsDistanceType getCuvsDistanceType() {
    return cuvsDistanceType;
  }

  /**
   * get the number of Iterations to run if building with NN_DESCENT
   *
   * @return the number of iterations for NN_DESCENT
   */
  public int getNNDescentNumIterations() {
    return nnDescentNumIterations;
  }

  @Override
  public String toString() {
    return "ParamsBase [writerThreads="
        + writerThreads
        + ", intermediateGraphDegree="
        + intermediateGraphDegree
        + ", graphdegree="
        + graphdegree
        + ", maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", cagraGraphBuildAlgo="
        + cagraGraphBuildAlgo
        + ", cuVSIvfPqParams="
        + cuVSIvfPqParams
        + ", strategy="
        + strategy
        + ", cuvsDistanceType="
        + cuvsDistanceType
        + ", nnDescentNumIterations="
        + nnDescentNumIterations
        + "]";
  }

  /**
   * Validates the base input parameters.
   *
   * @throws IllegalArgumentException
   */
  private void validate() throws IllegalArgumentException {
    if (writerThreads < MIN_WRITER_THREADS || writerThreads > MAX_WRITER_THREADS) {
      throw new IllegalArgumentException(
          "writerThreads not in valid range. Valid range: ["
              + MIN_WRITER_THREADS
              + ", "
              + MAX_WRITER_THREADS
              + "]");
    }
    if (intermediateGraphDegree < MIN_INT_GRAPH_DEG
        || intermediateGraphDegree > MAX_INT_GRAPH_DEG) {
      throw new IllegalArgumentException(
          "intermediateGraphDegree not in valid range. Valid range: ["
              + MIN_INT_GRAPH_DEG
              + ", "
              + MAX_INT_GRAPH_DEG
              + "]");
    }
    if (graphdegree < MIN_GRAPH_DEG || graphdegree > MAX_GRAPH_DEG) {
      throw new IllegalArgumentException(
          "graphdegree not in valid range. Valid range: ["
              + MIN_GRAPH_DEG
              + ", "
              + MAX_GRAPH_DEG
              + "]");
    }
    if (maxConn < MIN_MAX_CONN || maxConn > MAX_MAX_CONN) {
      throw new IllegalArgumentException(
          "maxConn not in valid range. Valid range: [" + MIN_MAX_CONN + ", " + MAX_MAX_CONN + "]");
    }
    if (beamWidth < MIN_BEAM_WIDTH || beamWidth > MAX_BEAM_WIDTH) {
      throw new IllegalArgumentException(
          "beamWidth not in valid range. Valid range: ["
              + MIN_BEAM_WIDTH
              + ", "
              + MAX_BEAM_WIDTH
              + "]");
    }
    if (Objects.isNull(cagraGraphBuildAlgo)) {
      throw new IllegalArgumentException("cagraGraphBuildAlgo cannot be null.");
    }
    if (Objects.isNull(strategy)) {
      throw new IllegalArgumentException("strategy cannot be null.");
    }
    if (Objects.isNull(cuvsDistanceType)) {
      throw new IllegalArgumentException("cuvsDistanceType cannot be null.");
    }
    if (nnDescentNumIterations < MIN_NN_DESCENT_NUM_ITERATIONS
        || nnDescentNumIterations > MAX_NN_DESCENT_NUM_ITERATIONS) {
      throw new IllegalArgumentException(
          "nnDescentNumIterations not in valid range. Valid range: ["
              + MIN_NN_DESCENT_NUM_ITERATIONS
              + ", "
              + MAX_NN_DESCENT_NUM_ITERATIONS
              + "]");
    }
  }
}
