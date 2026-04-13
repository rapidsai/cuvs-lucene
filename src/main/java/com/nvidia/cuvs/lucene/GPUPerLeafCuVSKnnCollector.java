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
  private CagraSearchParams.SearchAlgo searchAlgo;

  /**
   * Initializes {@link GPUPerLeafCuVSKnnCollector}
   *
   * @param topK the topK value
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   * @param searchAlgo the CAGRA search algorithm
   */
  public GPUPerLeafCuVSKnnCollector(
      int topK,
      int visitLimit,
      int iTopK,
      int searchWidth,
      CagraSearchParams.SearchAlgo searchAlgo) {
    super(topK, visitLimit);
    this.iTopK = iTopK > topK ? iTopK : topK;
    this.searchWidth = searchWidth;
    this.searchAlgo = searchAlgo;
  }

  public int getiTopK() {
    return iTopK;
  }

  public int getSearchWidth() {
    return searchWidth;
  }

  public CagraSearchParams.SearchAlgo getSearchAlgo() {
    return searchAlgo;
  }
}
