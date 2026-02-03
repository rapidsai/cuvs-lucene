/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AcceleratedHNSWParams {

  /*
   * TODO: Update boundaries for all parameters when a consensus is reached.
   * Issue: https://github.com/rapidsai/cuvs-lucene/issues/99
   */
  private static final int MIN_WRITER_THREADS = 1;
  private static final int MAX_WRITER_THREADS = 32;
  private static final int MIN_INT_GRAPH_DEG = 2;
  private static final int MAX_INT_GRAPH_DEG = 128;
  private static final int MIN_GRAPH_DEG = 1;
  private static final int MAX_GRAPH_DEG = 64;
  private static final int MIN_HNSW_LAYERS = 1;
  private static final int MAX_HNSW_LAYERS = 5;
  private static final int MIN_MAX_CONN = 1;
  private static final int MAX_MAX_CONN = 512;
  private static final int MIN_BEAM_WIDTH = 1;
  private static final int MAX_BEAM_WIDTH = 512;
  private static final int MIN_NUM_MERGE_WORKERS = 1;
  private static final int MAX_NUM_MERGE_WORKERS = 32;

  private final int writerThreads;
  private final int intermediateGraphDegree;
  private final int graphdegree;
  private final int hnswLayers;
  private final int maxConn;
  private final int beamWidth;
  private final int numMergeWorkers;
  private final ExecutorService mergeExec;

  /**
   * Constructs an instance of {@link GPUSearchParams} with specific parameter values.
   *
   * @param writerThreads Number of cuVS writer threads to use.
   * @param intermediateGraphDegree The intermediate graph degree while building the CAGRA index.
   * @param graphdegree The graph degree to use while building the CAGRA index.
   * @param indexType The type of index to build - CAGRA, BRUTEFORCE, or both.
   * @param hnswLayers The number of HNSW layers to build in the HNSW index.
   * @param maxConn The max connection parameter used when building HNSW index with the fallback mechanism.
   * @param beamWidth The beam width parameter used when building HNSW index with the fallback mechanism.
   * @param numMergeWorkers The number of merge workers to use with the fallback mechanism.
   * @param mergeExec The instance of {@link ExecutorService} to use with the fallback mechanism.
   */
  private AcceleratedHNSWParams(
      int writerThreads,
      int intermediateGraphDegree,
      int graphdegree,
      int hnswLayers,
      int maxConn,
      int beamWidth,
      int numMergeWorkers,
      ExecutorService mergeExec) {
    super();
    this.writerThreads = writerThreads;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphdegree = graphdegree;
    this.hnswLayers = hnswLayers;
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
    this.numMergeWorkers = numMergeWorkers;
    this.mergeExec = mergeExec;
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
   * Get the number of HNSW layers
   *
   * @return the number of HNSW layers
   */
  public int getHnswLayers() {
    return hnswLayers;
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
   * Get the number of merge workers set to be used in the fallback mechanism
   *
   * @return the number of merge workers
   */
  public int getNumMergeWorkers() {
    return numMergeWorkers;
  }

  /**
   * Get the instance of the {@link ExecutorService} to be used in the fallback mechanism   *
   *
   * @return the instance of the {@link ExecutorService}
   */
  public ExecutorService getMergeExec() {
    return mergeExec;
  }

  @Override
  public String toString() {
    return "AcceleratedHNSWParams [writerThreads="
        + writerThreads
        + ", intermediateGraphDegree="
        + intermediateGraphDegree
        + ", graphdegree="
        + graphdegree
        + ", hnswLayers="
        + hnswLayers
        + ", maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", numMergeWorkers="
        + numMergeWorkers
        + "]";
  }

  /**
   * Builder class for creating an instance of {@link GPUSearchParams}
   */
  public static class Builder {

    private int writerThreads = 1;
    private int intermediateGraphDegree = 128;
    private int graphdegree = 64;
    private int hnswLayers = 2;
    private int maxConn = 8;
    private int beamWidth = 16;
    private int numMergeWorkers = 1;
    private ExecutorService mergeExec = Executors.newFixedThreadPool(1);

    /**
     * Set the number of cuVS writer threads while building the index
     * Valid range - Minimum: {@value MIN_WRITER_THREADS}, Maximum: {@value MAX_WRITER_THREADS}
     * Default value - 64
     *
     * @param writerThreads
     * @return instance of {@link Builder}
     */
    public Builder withWriterThreads(int writerThreads) {
      this.writerThreads = writerThreads;
      return this;
    }

    /**
     * Set the intermediate graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_INT_GRAPH_DEG}, Maximum: {@value MAX_INT_GRAPH_DEG}
     * Default value - 128
     *
     * @param intermediateGraphDegree
     * @return instance of {@link Builder}
     */
    public Builder withIntermediateGraphDegree(int intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    /**
     * Set the graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_GRAPH_DEG}, Maximum: {@value MAX_GRAPH_DEG}
     * Default value - 64
     *
     * @param graphDegree
     * @return instance of {@link Builder}
     */
    public Builder withGraphDegree(int graphDegree) {
      this.graphdegree = graphDegree;
      return this;
    }

    /**
     * Set the number of HNSW layers to construct while building the HNSW index
     * Valid range - Minimum: {@value MIN_HNSW_LAYERS}, Maximum: {@value MAX_HNSW_LAYERS}
     * Default value - 2
     *
     * @param hnswLayers the number of HNSW layers
     * @return instance of {@link Builder}
     */
    public Builder withHNSWLayer(int hnswLayers) {
      this.hnswLayers = hnswLayers;
      return this;
    }

    /**
     * Set the max connections parameter while building HNSW index with fallback mechanism
     * Valid range - Minimum: {@value MIN_MAX_CONN}, Maximum: {@value MAX_MAX_CONN}
     * Default value - 8
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
     * Default value - 16
     *
     * @param beamWidth the beam width parameter
     * @return instance of {@link Builder}
     */
    public Builder withBeamWidth(int beamWidth) {
      this.beamWidth = beamWidth;
      return this;
    }

    /**
     * Set the number of merge workers to be used with the fallback mechanism
     * Default value - 1
     *
     * @param numMergeWorkers number of merge workers to set
     * @return instance of {@link Builder}
     */
    public Builder withNumMergeWorkers(int numMergeWorkers) {
      this.numMergeWorkers = numMergeWorkers;
      return this;
    }

    /**
     * Set the merge executor service to be used in the fallback mechanism
     * Default value an instance with one thread
     *
     * @param mergeExec an instance of {@link ExecutorService}
     * @return instance of {@link Builder}
     */
    public Builder withMergeExecutorService(ExecutorService mergeExec) {
      this.mergeExec = mergeExec;
      return this;
    }

    /**
     * Validates the input parameters.
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
      if (hnswLayers < MIN_HNSW_LAYERS || hnswLayers > MAX_HNSW_LAYERS) {
        throw new IllegalArgumentException(
            "hnswLayers not in valid range. Valid range: ["
                + MIN_HNSW_LAYERS
                + ", "
                + MAX_HNSW_LAYERS
                + "]");
      }
      if (maxConn < MIN_MAX_CONN || maxConn > MAX_MAX_CONN) {
        throw new IllegalArgumentException(
            "maxConn not in valid range. Valid range: ["
                + MIN_MAX_CONN
                + ", "
                + MAX_MAX_CONN
                + "]");
      }
      if (beamWidth < MIN_BEAM_WIDTH || beamWidth > MAX_BEAM_WIDTH) {
        throw new IllegalArgumentException(
            "beamWidth not in valid range. Valid range: ["
                + MIN_BEAM_WIDTH
                + ", "
                + MAX_BEAM_WIDTH
                + "]");
      }
      if (numMergeWorkers < MIN_NUM_MERGE_WORKERS || numMergeWorkers > MAX_NUM_MERGE_WORKERS) {
        throw new IllegalArgumentException(
            "numMergeWorkers not in valid range. Valid range: ["
                + MIN_NUM_MERGE_WORKERS
                + ", "
                + MAX_NUM_MERGE_WORKERS
                + "]");
      }
      if (Objects.isNull(mergeExec)) {
        throw new IllegalArgumentException("mergeExec cannot be null.");
      }
    }

    /**
     * Create an instance of {@link AcceleratedHNSWParams}
     *
     * @return instance of {@link AcceleratedHNSWParams}
     */
    public AcceleratedHNSWParams build() {
      validate();
      return new AcceleratedHNSWParams(
          writerThreads,
          intermediateGraphDegree,
          graphdegree,
          hnswLayers,
          maxConn,
          beamWidth,
          numMergeWorkers,
          mergeExec);
    }
  }
}
