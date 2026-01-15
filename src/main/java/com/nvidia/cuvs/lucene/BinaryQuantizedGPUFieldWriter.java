/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorEncoding.FLOAT32;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.RamUsageEstimator;

public class BinaryQuantizedGPUFieldWriter extends KnnFieldVectorsWriter<Object> {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(BinaryQuantizedGPUFieldWriter.class);

  private final FieldInfo fieldInfo;
  private final FlatFieldVectorsWriter<?> flatFieldVectorsWriter;
  private final boolean isFloatEncoding;
  private int lastDocID = -1;

  @SuppressWarnings("unchecked")
  public BinaryQuantizedGPUFieldWriter(
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
      return quantizeFloatVectorsToBinary(floatVectors);
    } else {
      @SuppressWarnings("unchecked")
      FlatFieldVectorsWriter<byte[]> byteWriter =
          (FlatFieldVectorsWriter<byte[]>) flatFieldVectorsWriter;
      return byteWriter.getVectors();
    }
  }

  /**
   * Quantizes FLOAT32 vectors to binary (1 bit per dimension, packed into bytes).
   * Binary quantization: each dimension is compared to a centroid (mean of all values for that dimension).
   * If value > centroid, bit = 1, else bit = 0.
   * Bits are packed: 8 dimensions per byte.
   */
  private List<byte[]> quantizeFloatVectorsToBinary(List<float[]> floatVectors) {
    if (floatVectors.isEmpty()) {
      return new ArrayList<>();
    }

    int dimensions = floatVectors.get(0).length;
    int numVectors = floatVectors.size();
    int bytesPerVector = (dimensions + 7) / 8;

    float[] centroids = new float[dimensions];
    for (float[] vector : floatVectors) {
      for (int d = 0; d < dimensions; d++) {
        centroids[d] += vector[d];
      }
    }
    for (int d = 0; d < dimensions; d++) {
      centroids[d] /= numVectors;
    }

    List<byte[]> quantizedVectors = new ArrayList<>(numVectors);
    for (float[] vector : floatVectors) {
      byte[] quantized = new byte[bytesPerVector];
      for (int d = 0; d < dimensions; d++) {
        boolean bit = vector[d] > centroids[d];
        int byteIndex = d / 8;
        int bitIndex = d % 8;
        if (bit) {
          quantized[byteIndex] |= (1 << bitIndex);
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
