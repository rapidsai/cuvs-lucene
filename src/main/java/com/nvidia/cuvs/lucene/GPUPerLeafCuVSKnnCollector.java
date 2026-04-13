/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraSearchParams;
import org.apache.lucene.search.TopKnnCollector;

/**
 * KnnCollector for cuVS used for search on the GPU.
 *
 * @since 25.10
 */
class GPUPerLeafCuVSKnnCollector extends TopKnnCollector {

  private int iTopK;
  private int searchWidth;
  private int threadBlockSize;
  private CagraSearchParams.SearchAlgo searchAlgo;
  private boolean persistent;
  private float persistentLifetime;
  private float persistentDeviceUsage;

  /**
   * Initializes {@link GPUPerLeafCuVSKnnCollector}
   *
   * @param topK the topK value
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   * @param threadBlockSize CAGRA thread_block_size (0 = auto; controls worker_queue_size)
   * @param searchAlgo the CAGRA search algorithm
   * @param persistent whether to use the persistent kernel
   * @param persistentLifetime persistent kernel lifetime in seconds
   * @param persistentDeviceUsage fraction of GPU SMs for the persistent kernel
   */
  public GPUPerLeafCuVSKnnCollector(
      int topK,
      int visitLimit,
      int iTopK,
      int searchWidth,
      int threadBlockSize,
      CagraSearchParams.SearchAlgo searchAlgo,
      boolean persistent,
      float persistentLifetime,
      float persistentDeviceUsage) {
    super(topK, visitLimit);
    this.iTopK = iTopK > topK ? iTopK : topK;
    this.searchWidth = searchWidth;
    this.threadBlockSize = threadBlockSize;
    this.searchAlgo = searchAlgo;
    this.persistent = persistent;
    this.persistentLifetime = persistentLifetime;
    this.persistentDeviceUsage = persistentDeviceUsage;
  }

  public int getiTopK() {
    return iTopK;
  }

  public int getSearchWidth() {
    return searchWidth;
  }

  public int getThreadBlockSize() {
    return threadBlockSize;
  }

  public CagraSearchParams.SearchAlgo getSearchAlgo() {
    return searchAlgo;
  }

  public boolean isPersistent() {
    return persistent;
  }

  public float getPersistentLifetime() {
    return persistentLifetime;
  }

  public float getPersistentDeviceUsage() {
    return persistentDeviceUsage;
  }
}
