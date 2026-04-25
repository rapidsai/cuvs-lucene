/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.CuVSIvfPqSearchParams;

public class CagraIndexParamsFactory {

  private static final int ALGO_SWITCH_THRESHOLD = 5_000_000;

  private interface IndexParams {
    CagraIndexParams get();
  }

  private class NNDescentParams implements IndexParams {

    private int graphDegree;
    private int intGraphDegree;
    private int writerThreads;
    private int nnDescentNumIterations;
    private CuvsDistanceType cuvsDistanceType;

    private NNDescentParams(
        int graphDegree,
        int intGraphDegree,
        int writerThreads,
        int nnDescentNumIterations,
        CuvsDistanceType cuvsDistanceType) {
      super();
      this.graphDegree = graphDegree;
      this.intGraphDegree = intGraphDegree;
      this.writerThreads = writerThreads;
      this.nnDescentNumIterations = nnDescentNumIterations;
      this.cuvsDistanceType = cuvsDistanceType;
    }

    @Override
    public CagraIndexParams get() {
      return new CagraIndexParams.Builder()
          .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
          .withGraphDegree(graphDegree)
          .withIntermediateGraphDegree(intGraphDegree)
          .withNNDescentNumIterations(nnDescentNumIterations)
          .withNumWriterThreads(writerThreads)
          .withMetric(cuvsDistanceType)
          .build();
    }
  }

  private class IVFPQIndexParams implements IndexParams {

    private long rows;
    private long dimension;
    private int graphDegree;
    private int intGraphDegree;
    private int writerThreads;

    private IVFPQIndexParams(
        long rows, long dimension, int graphDegree, int intGraphDegree, int writerThreads) {
      super();
      this.rows = rows;
      this.dimension = dimension;
      this.graphDegree = graphDegree;
      this.intGraphDegree = intGraphDegree;
      this.writerThreads = writerThreads;
    }

    @Override
    public CagraIndexParams get() {
      return new CagraIndexParams.Builder()
          .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
          .withCuVSIvfPqParams(getCuVSIvfPqParams(rows, dimension))
          .withNumWriterThreads(writerThreads)
          .withIntermediateGraphDegree(intGraphDegree)
          .withGraphDegree(graphDegree)
          .build();
    }

    private CuVSIvfPqParams getCuVSIvfPqParams(long rows, long dimension) {

      int pqDim;
      int pqBits;

      if (dimension <= 32) {
        pqDim = 16;
        pqBits = 8;
      } else {
        pqBits = 4;
        if (dimension <= 64) {
          pqDim = 32;
        } else if (dimension <= 128) {
          pqDim = 64;
        } else if (dimension <= 192) {
          pqDim = 96;
        } else {
          pqDim = (int) Math.round(Math.ceil(dimension / 2));
        }
      }

      int nLists = (int) Math.max(1, rows / 2000);
      int kmeansNIters = 10;

      double kMinPointsPerCluster = 32;
      double minKmeansPrainsetPoints = kMinPointsPerCluster * nLists;
      double maxKmeansTrainsetFraction = 1.0;
      double minKmeansTrainsetFraction =
          Math.min(maxKmeansTrainsetFraction, minKmeansPrainsetPoints / rows);
      double kmeansTrainsetFraction =
          Math.clamp(
              1.0 / Math.sqrt(rows * 1e-5), minKmeansTrainsetFraction, maxKmeansTrainsetFraction);

      CodebookGen codebookKind = CodebookGen.PER_SUBSPACE;

      int nProbes = (int) Math.round(Math.sqrt(nLists) / 20 + 4);
      // search_params.coarse_search_dtype = CUDA_R_16F;
      // max_internal_batch_size = 128 * 1024;
      int refinementRate = 1;

      CuVSIvfPqIndexParams cuVSIvfPqIndexParams =
          new CuVSIvfPqIndexParams.Builder()
              .withCodebookKind(codebookKind)
              .withKmeansNIters(kmeansNIters)
              .withKmeansTrainsetFraction(kmeansTrainsetFraction)
              .withNLists(nLists)
              .withPqBits(pqBits)
              .withPqDim(pqDim)
              .build();

      CuVSIvfPqSearchParams cuVSIvfPqSearchParams =
          new CuVSIvfPqSearchParams.Builder()
              .withLutDtype(CudaDataType.CUDA_R_16F)
              .withInternalDistanceDtype(CudaDataType.CUDA_R_16F)
              .withNProbes(nProbes)
              .build();

      CuVSIvfPqParams cuVSIvfPqParams =
          new CuVSIvfPqParams.Builder()
              .withCuVSIvfPqIndexParams(cuVSIvfPqIndexParams)
              .withCuVSIvfPqSearchParams(cuVSIvfPqSearchParams)
              .withRefinementRate(refinementRate)
              .build();

      return cuVSIvfPqParams;
    }
  }

  public CagraIndexParams create(GPUSearchParams params, long rows, long dimension) {
    if (params.getStrategy().equals(GPUSearchParams.Strategy.HEURISTIC)) {
      if (rows < ALGO_SWITCH_THRESHOLD) {
        return new NNDescentParams(
                params.getGraphdegree(),
                params.getIntermediateGraphDegree(),
                params.getWriterThreads(),
                20,
                params.getCuvsDistanceType())
            .get();
      } else {
        return new IVFPQIndexParams(
                rows,
                dimension,
                params.getGraphdegree(),
                params.getIntermediateGraphDegree(),
                params.getWriterThreads())
            .get();
      }
    } else {
      return new CagraIndexParams.Builder()
          .withNumWriterThreads(params.getWriterThreads())
          .withIntermediateGraphDegree(params.getIntermediateGraphDegree())
          .withGraphDegree(params.getGraphdegree())
          .withCagraGraphBuildAlgo(params.getCagraGraphBuildAlgo())
          .withCuVSIvfPqParams(params.getCuVSIvfPqParams())
          .build();
    }
  }

  public CagraIndexParams create(AcceleratedHNSWParams params, long rows, long dimension) {
    if (params.getStrategy().equals(AcceleratedHNSWParams.Strategy.HEURISTIC)) {
      if (rows < ALGO_SWITCH_THRESHOLD) {
        return new NNDescentParams(
                params.getGraphdegree(),
                params.getIntermediateGraphDegree(),
                params.getWriterThreads(),
                20,
                params.getCuvsDistanceType())
            .get();
      } else {
        return new IVFPQIndexParams(
                rows,
                dimension,
                params.getGraphdegree(),
                params.getIntermediateGraphDegree(),
                params.getWriterThreads())
            .get();
      }
    } else {
      return new CagraIndexParams.Builder()
          .withNumWriterThreads(params.getWriterThreads())
          .withIntermediateGraphDegree(params.getIntermediateGraphDegree())
          .withGraphDegree(params.getGraphdegree())
          .withCagraGraphBuildAlgo(params.getCagraGraphBuildAlgo())
          .withCuVSIvfPqParams(params.getCuVSIvfPqParams())
          .build();
    }
  }
}
