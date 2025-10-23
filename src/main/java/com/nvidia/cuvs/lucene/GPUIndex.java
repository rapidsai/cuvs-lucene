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

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.CagraIndex;
import java.io.Closeable;
import java.io.IOException;
import java.util.Objects;

/**
 * This class holds references to the actual cuVS Index (Cagra, Brute force, etc.)
 *
 * @since 25.10
 */
public class GPUIndex implements Closeable {
  private final CagraIndex cagraIndex;
  private final BruteForceIndex bruteforceIndex;

  private int maxDocs;
  private String fieldName;
  private String segmentName;
  private volatile boolean closed;

  /**
   * Initializes an instance of {@link GPUIndex}
   *
   * @param segmentName the name of the segment
   * @param fieldName the field name
   * @param cagraIndex reference to the CagraIndex
   * @param maxDocs the maximum documents
   * @param bruteforceIndex reference to the BruteForceIndex
   */
  public GPUIndex(
      String segmentName,
      String fieldName,
      CagraIndex cagraIndex,
      int maxDocs,
      BruteForceIndex bruteforceIndex) {
    this.cagraIndex = Objects.requireNonNull(cagraIndex);
    this.bruteforceIndex = Objects.requireNonNull(bruteforceIndex);
    this.fieldName = Objects.requireNonNull(fieldName);
    this.segmentName = Objects.requireNonNull(segmentName);
    if (maxDocs < 0) {
      throw new IllegalArgumentException("negative maxDocs:" + maxDocs);
    }
    this.maxDocs = maxDocs;
  }

  /**
   * Initializes an instance of {@link GPUIndex}
   * @param cagraIndex reference to the CagraIndex instance
   * @param bruteforceIndex reference to the instances of BruteForceIndex
   */
  public GPUIndex(CagraIndex cagraIndex, BruteForceIndex bruteforceIndex) {
    this.cagraIndex = cagraIndex;
    this.bruteforceIndex = bruteforceIndex;
  }

  /**
   * Gets the reference to CAGRA index
   * @return an instance of CagraIndex
   */
  public CagraIndex getCagraIndex() {
    ensureOpen();
    return cagraIndex;
  }

  /**
   * Gets the reference to the Bruteforce index
   * @return an instance of BruteForceIndex
   */
  public BruteForceIndex getBruteforceIndex() {
    ensureOpen();
    return bruteforceIndex;
  }

  /**
   * Gets the field name
   * @return field name
   */
  public String getFieldName() {
    return fieldName;
  }

  /**
   * Gets the segment name
   *
   * @return segment name
   */
  public String getSegmentName() {
    return segmentName;
  }

  /**
   * Gets the max docs
   * @return the max docs
   */
  public int getMaxDocs() {
    return maxDocs;
  }

  /**
   * Throws {@link IllegalArgumentException} if the index is closed
   */
  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("index is closed");
    }
  }

  /**
   * Closes this stream and releases any system resources associated with it. If the stream is already closed then invoking this method has no effect.
   */
  @Override
  public void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    destroyIndices();
  }

  /**
   * Closes the cuVS indexes.
   *
   * @throws IOException
   */
  private void destroyIndices() throws IOException {
    try {
      if (cagraIndex != null) {
        cagraIndex.close();
      }
      if (bruteforceIndex != null) {
        bruteforceIndex.close();
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }
}
