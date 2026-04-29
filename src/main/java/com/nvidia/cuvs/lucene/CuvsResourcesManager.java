/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo.IVF_PQ;
import static com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.GPUInfoProvider;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

public class CuvsResourcesManager {

  private static final Logger LOG = Logger.getLogger(Utils.class.getName());
  private static final CuVSProvider PROVIDER = CuVSProvider.provider();
  private static final GPUInfoProvider GPU_INFO_PROVIDER = PROVIDER.gpuInfoProvider();
  private static final int MAX_POOL_SIZE = 512;

  private ManagedCuVSResources[] pool;
  private ReentrantLock lock;
  private Condition resourcesAvailable;
  private AtomicLong reserveMemory;
  private long totalDeviceMemory;
  private int capacity;

  public CuvsResourcesManager(int capacity) {
    if (capacity > MAX_POOL_SIZE || capacity <= 0) {
      throw new IllegalArgumentException(
          "Invalid capacity, should be between 1 and " + MAX_POOL_SIZE);
    }
    this.capacity = capacity;
    pool = new ManagedCuVSResources[capacity];
    lock = new ReentrantLock();
    resourcesAvailable = lock.newCondition();
    for (int i = 0; i < capacity; i++) {
      pool[i] = new ManagedCuVSResources(getCuVSResourceInstance());
    }
    CuVSResources rx = getCuVSResourceInstance();
    reserveMemory = new AtomicLong();
    totalDeviceMemory = GPU_INFO_PROVIDER.getCurrentInfo(rx).totalDeviceMemoryInBytes();
    rx.close();
  }

  private ManagedCuVSResources getAvailableResourcesFromPool() {
    return Arrays.stream(pool).filter(mcr -> !mcr.isLocked()).findFirst().orElse(null);
  }

  private long getNumLockedResources() {
    return Arrays.stream(pool).filter(mcr -> mcr.isLocked()).count();
  }

  public ManagedCuVSResources acquireResource(long rows, long dimension, CagraIndexParams params)
      throws InterruptedException {
    try {
      lock.lock();
      long neededMemory = getEstimatedMemoryRequirement(rows, dimension, params);

      if (neededMemory > totalDeviceMemory) {
        throw new RuntimeException("Not enough GPU device memory available");
      }

      while (getNumLockedResources() == capacity
          || (totalDeviceMemory - reserveMemory.get()) < neededMemory) {
        resourcesAvailable.await();
      }
      reserveMemory.addAndGet(neededMemory);

      ManagedCuVSResources managedCuVSResources = getAvailableResourcesFromPool();
      assert managedCuVSResources != null;

      managedCuVSResources.setNeededMemory(neededMemory);
      managedCuVSResources.lock();
      return managedCuVSResources;
    } finally {
      lock.unlock();
    }
  }

  public void releaseResource(ManagedCuVSResources resource) {
    try {
      lock.lock();
      reserveMemory.addAndGet(-resource.getNeededMemory());
      resource.unlock();
      resourcesAvailable.signalAll();
    } finally {
      lock.unlock();
    }
  }

  public void shutdown() {
    Arrays.stream(pool)
        .forEach(
            mcr -> {
              if (Objects.nonNull(mcr) && Objects.nonNull(mcr.getResource())) {
                mcr.getResource().close();
              }
            });
  }

  private static CuVSResources getCuVSResourceInstance() {
    try {
      return CuVSResources.create();
    } catch (UnsupportedOperationException uoe) {
      LOG.log(
          Level.WARNING,
          "cuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      if (t instanceof ExceptionInInitializerError ex) {
        t = ex.getCause();
      }
      LOG.log(Level.WARNING, "Exception occurred during creation of cuVS resources. " + t);
    }
    return null;
  }

  private long getEstimatedMemoryRequirement(long rows, long dimension, CagraIndexParams params) {
    CagraGraphBuildAlgo buildAlgo = params.getCagraGraphBuildAlgo();
    if (buildAlgo.equals(NN_DESCENT)) {
      return 2 * rows * dimension * Float.BYTES;
    } else if (buildAlgo.equals(IVF_PQ)) {
      assert params.getCuVSIvfPqParams() != null;
      CuVSIvfPqIndexParams ip = params.getCuVSIvfPqParams().getIndexParams();
      assert ip != null;
      return 2
          * (long)
              (rows * (ip.getPqDim() * (ip.getPqBits() / 8.0) + Float.BYTES)
                  + ip.getnLists() * Integer.BYTES);
    } else {
      throw new IllegalArgumentException("Unsupported CAGRA build algo");
    }
  }

  /**
   * Holds reference to CuVSResources with its associated lock, and needed memory.
   */
  class ManagedCuVSResources {

    private final CuVSResources cuVSResources;
    private final ReentrantLock lock;
    private long neededMemory;

    public ManagedCuVSResources(CuVSResources cuVSResources) {
      this.cuVSResources = cuVSResources;
      lock = new ReentrantLock();
    }

    public CuVSResources getResource() {
      return cuVSResources;
    }

    public long getNeededMemory() {
      return neededMemory;
    }

    public void setNeededMemory(long neededMemory) {
      this.neededMemory = neededMemory;
    }

    public void lock() {
      lock.lock();
    }

    public void unlock() {
      lock.unlock();
    }

    public boolean isLocked() {
      return lock.isLocked();
    }
  }
}
