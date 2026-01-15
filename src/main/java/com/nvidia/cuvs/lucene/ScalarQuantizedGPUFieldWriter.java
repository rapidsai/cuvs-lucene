/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorEncoding.FLOAT32;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;

/**
 * Helper class for scalar quantized field writer
 * Handles both FLOAT32 (quantizes to 7-bit signed bytes) and BYTE (already quantized) encodings
 */
public class ScalarQuantizedGPUFieldWriter extends KnnFieldVectorsWriter<Object> {

  private static final long SHALLOW_SIZE =
      org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance(
          ScalarQuantizedGPUFieldWriter.class);

  private final FieldInfo fieldInfo;
  private final FlatFieldVectorsWriter<?> flatFieldVectorsWriter;
  private final boolean isFloatEncoding;
  private int lastDocID = -1;

  @SuppressWarnings("unchecked")
  public ScalarQuantizedGPUFieldWriter(
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
      return quantizeFloatVectors(floatVectors);
    } else {
      @SuppressWarnings("unchecked")
      FlatFieldVectorsWriter<byte[]> byteWriter =
          (FlatFieldVectorsWriter<byte[]>) flatFieldVectorsWriter;
      return byteWriter.getVectors();
    }
  }

  private List<byte[]> quantizeFloatVectors(List<float[]> floatVectors) {
    if (floatVectors.isEmpty()) {
      return new ArrayList<>();
    }

    int dimensions = floatVectors.get(0).length;
    int numVectors = floatVectors.size();

    float[] minPerDim = new float[dimensions];
    float[] maxPerDim = new float[dimensions];
    Arrays.fill(minPerDim, Float.MAX_VALUE);
    Arrays.fill(maxPerDim, Float.MIN_VALUE);

    for (float[] vector : floatVectors) {
      for (int d = 0; d < dimensions; d++) {
        minPerDim[d] = Math.min(minPerDim[d], vector[d]);
        maxPerDim[d] = Math.max(maxPerDim[d], vector[d]);
      }
    }

    List<byte[]> quantizedVectors = new ArrayList<>(numVectors);
    for (float[] vector : floatVectors) {
      byte[] quantized = new byte[dimensions];
      for (int d = 0; d < dimensions; d++) {
        float range = maxPerDim[d] - minPerDim[d];
        if (range > 0) {
          float normalized = (vector[d] - minPerDim[d]) / range;
          int quantizedValue = Math.round(normalized * 127.0f) - 64;
          quantized[d] = (byte) Math.max(-64, Math.min(63, quantizedValue));
        } else {
          quantized[d] = 0;
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
