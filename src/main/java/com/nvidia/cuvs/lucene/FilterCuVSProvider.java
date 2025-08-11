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
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.TieredIndex;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/*package-private*/ class FilterCuVSProvider implements CuVSProvider {

  private final CuVSProvider delegate;

  FilterCuVSProvider(CuVSProvider delegate) {
    this.delegate = delegate;
  }

  @Override
  public Path nativeLibraryPath() {
    return CuVSProvider.TMPDIR;
  }

  @Override
  public CuVSResources newCuVSResources(Path tempPath) throws Throwable {
    return delegate.newCuVSResources(tempPath);
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newBruteForceIndexBuilder(cuVSResources);
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newCagraIndexBuilder(cuVSResources);
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newHnswIndexBuilder(cuVSResources);
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] arg0) throws Throwable {
    return delegate.mergeCagraIndexes(arg0);
  }

  @Override
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newTieredIndexBuilder(cuVSResources);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] array) {
    return delegate.newMatrixFromArray(array);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] array) {
    return delegate.newMatrixFromArray(array);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] array) {
    return delegate.newMatrixFromArray(array);
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    return delegate.newNativeMatrixBuilder();
  }

  @Override
  public CuVSMatrix.Builder newMatrixBuilder(int rows, int cols, CuVSMatrix.DataType dataType) {
    return delegate.newMatrixBuilder(rows, cols, dataType);
  }
}
