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

  private static CuVSIvfPqParams getCuVSIvfPqParams(long rows, long dimension) {

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
        pqDim = (int) roundUpSafe(dimension / 2, 128);
      }
    }

    int nLists = (int) Math.max(1, rows / 2000);
    final int kmeansNIters = 10;
    final double kMinPointsPerCluster = 32;
    double minKmeansPrainsetPoints = kMinPointsPerCluster * nLists;
    final double maxKmeansTrainsetFraction = 1.0;
    double minKmeansTrainsetFraction =
        Math.min(maxKmeansTrainsetFraction, minKmeansPrainsetPoints / rows);
    double kmeansTrainsetFraction =
        Math.clamp(
            1.0 / Math.sqrt(rows * 1e-5), minKmeansTrainsetFraction, maxKmeansTrainsetFraction);
    final CodebookGen codebookKind = CodebookGen.PER_SUBSPACE;
    int nProbes = (int) Math.round(Math.sqrt(nLists) / 20 + 4);
    final int refinementRate = 1;

    CuVSIvfPqIndexParams cuVSIvfPqIndexParams =
        new CuVSIvfPqIndexParams.Builder()
            .withCodebookKind(codebookKind)
            .withKmeansNIters(kmeansNIters)
            .withKmeansTrainsetFraction(kmeansTrainsetFraction)
            .withNLists(nLists)
            .withPqBits(pqBits)
            .withPqDim(pqDim)
            .withAddDataOnBuild(false)
            .withConservativeMemoryAllocation(true)
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

  private static long roundUpSafe(long numberToRound, long modulus) {
    long remainder = numberToRound % modulus;
    if (remainder == 0) {
      return numberToRound;
    }
    long roundedUp = numberToRound - remainder + modulus;
    return roundedUp;
  }

  private static CagraIndexParams getNNDescentParams(
      int graphDegree,
      int intGraphDegree,
      int writerThreads,
      long nnDescentNumIterations,
      CuvsDistanceType cuvsDistanceType) {
    return new CagraIndexParams.Builder()
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .withGraphDegree(graphDegree)
        .withIntermediateGraphDegree(intGraphDegree)
        .withNNDescentNumIterations(nnDescentNumIterations)
        .withNumWriterThreads(writerThreads)
        .withMetric(cuvsDistanceType)
        .build();
  }

  private static CagraIndexParams getIVFPQParams(
      int graphDegree, int intGraphDegree, int writerThreads, long rows, long dimension) {
    return new CagraIndexParams.Builder()
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
        .withCuVSIvfPqParams(getCuVSIvfPqParams(rows, dimension))
        .withNumWriterThreads(writerThreads)
        .withIntermediateGraphDegree(intGraphDegree)
        .withGraphDegree(graphDegree)
        .build();
  }

  public static CagraIndexParams create(
      GPUSearchParams gPUSearchParams, long rows, long dimension) {
    if (gPUSearchParams.getStrategy().equals(GPUSearchParams.Strategy.HEURISTIC)) {
      if (rows < ALGO_SWITCH_THRESHOLD) {
        return getNNDescentParams(
            gPUSearchParams.getGraphdegree(),
            gPUSearchParams.getIntermediateGraphDegree(),
            gPUSearchParams.getWriterThreads(),
            gPUSearchParams.getnNDescentNumIterations(),
            gPUSearchParams.getCuvsDistanceType());
      } else {
        return getIVFPQParams(
            gPUSearchParams.getGraphdegree(),
            gPUSearchParams.getIntermediateGraphDegree(),
            gPUSearchParams.getWriterThreads(),
            rows,
            dimension);
      }
    } else {
      return new CagraIndexParams.Builder()
          .withNumWriterThreads(gPUSearchParams.getWriterThreads())
          .withIntermediateGraphDegree(gPUSearchParams.getIntermediateGraphDegree())
          .withGraphDegree(gPUSearchParams.getGraphdegree())
          .withCagraGraphBuildAlgo(gPUSearchParams.getCagraGraphBuildAlgo())
          .withCuVSIvfPqParams(gPUSearchParams.getCuVSIvfPqParams())
          .build();
    }
  }

  public static CagraIndexParams create(
      AcceleratedHNSWParams acceleratedHNSWParams, long rows, long dimension) {
    if (acceleratedHNSWParams.getStrategy().equals(AcceleratedHNSWParams.Strategy.HEURISTIC)) {
      if (rows < ALGO_SWITCH_THRESHOLD) {
        return getNNDescentParams(
            acceleratedHNSWParams.getGraphdegree(),
            acceleratedHNSWParams.getIntermediateGraphDegree(),
            acceleratedHNSWParams.getWriterThreads(),
            acceleratedHNSWParams.getnNDescentNumIterations(),
            acceleratedHNSWParams.getCuvsDistanceType());
      } else {
        return getIVFPQParams(
            acceleratedHNSWParams.getGraphdegree(),
            acceleratedHNSWParams.getIntermediateGraphDegree(),
            acceleratedHNSWParams.getWriterThreads(),
            rows,
            dimension);
      }
    } else {
      return new CagraIndexParams.Builder()
          .withNumWriterThreads(acceleratedHNSWParams.getWriterThreads())
          .withIntermediateGraphDegree(acceleratedHNSWParams.getIntermediateGraphDegree())
          .withGraphDegree(acceleratedHNSWParams.getGraphdegree())
          .withCagraGraphBuildAlgo(acceleratedHNSWParams.getCagraGraphBuildAlgo())
          .withCuVSIvfPqParams(acceleratedHNSWParams.getCuVSIvfPqParams())
          .build();
    }
  }
}
