/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.ConstraintViolationException;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import java.util.Set;
import org.hibernate.validator.messageinterpolation.ParameterMessageInterpolator;

public class GPUSearchParams {

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

  @Min(value = MIN_WRITER_THREADS)
  @Max(value = MAX_WRITER_THREADS)
  private final int writerThreads;

  @Min(value = MIN_INT_GRAPH_DEG)
  @Max(value = MAX_INT_GRAPH_DEG)
  private final int intermediateGraphDegree;

  @Min(value = MIN_GRAPH_DEG)
  @Max(value = MAX_GRAPH_DEG)
  private final int graphdegree;

  @NotNull private final CagraGraphBuildAlgo cagraGraphBuildAlgo;

  @NotNull private final IndexType indexType;

  /**
   * Constructs an instance of {@link GPUSearchParams} with specific parameter values.
   *
   * @param writerThreads Number of cuVS writer threads to use.
   * @param intermediateGraphDegree The intermediate graph degree while building the CAGRA index.
   * @param graphdegree The graph degree to use while building the CAGRA index.
   * @param cagraGraphBuildAlgo The CAGRA build algorithm to use.
   * @param indexType The type of index to build - CAGRA, BRUTEFORCE, or both.
   */
  private GPUSearchParams(
      int writerThreads,
      int intermediateGraphDegree,
      int graphdegree,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      IndexType indexType) {
    super();
    this.writerThreads = writerThreads;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphdegree = graphdegree;
    this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
    this.indexType = indexType;
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
   * Get the CAGRA build algorithm parameter value
   *
   * @return the CAGRA build algorithm parameter value
   */
  public CagraGraphBuildAlgo getCagraGraphBuildAlgo() {
    return cagraGraphBuildAlgo;
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
    return "GPUSearchParams [writerThreads="
        + writerThreads
        + ", intermediateGraphDegree="
        + intermediateGraphDegree
        + ", graphdegree="
        + graphdegree
        + ", cagraGraphBuildAlgo="
        + cagraGraphBuildAlgo
        + ", indexType="
        + indexType
        + "]";
  }

  /**
   * Builder class for creating an instance of {@link GPUSearchParams}
   */
  public static class Builder {

    private int writerThreads = 1;
    private int intermediateGraphDegree = 128;
    private int graphdegree = 64;
    private CagraGraphBuildAlgo cagraGraphBuildAlgo = CagraGraphBuildAlgo.NN_DESCENT;
    private IndexType indexType = IndexType.CAGRA;

    /**
     * Set the number of cuVS writer threads while building the index
     * Valid range - Minimum: {@value MIN_WRITER_THREADS}, Maximum: {@value MAX_WRITER_THREADS}
     * Default value - 64
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
     * Default value - 128
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
     * Default value - 64
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
     * Creates and returns an instance of {@link GPUSearchParams}
     *
     * @return instance of {@link GPUSearchParams}
     */
    public GPUSearchParams build() {
      ValidatorFactory factory =
          Validation.byDefaultProvider()
              .configure()
              .messageInterpolator(new ParameterMessageInterpolator())
              .buildValidatorFactory();
      Validator validator = factory.getValidator();

      GPUSearchParams gpuSearchParams =
          new GPUSearchParams(
              writerThreads, intermediateGraphDegree, graphdegree, cagraGraphBuildAlgo, indexType);
      Set<ConstraintViolation<GPUSearchParams>> violations = validator.validate(gpuSearchParams);

      if (!violations.isEmpty()) {
        throw new ConstraintViolationException(violations);
      }
      return gpuSearchParams;
    }
  }
}
