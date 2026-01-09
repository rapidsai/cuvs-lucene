/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSResources;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Provides a mechanism to create ThreadLocal based CuVSResource instances.
 *
 * @since 26.02
 */
public class CuVSResourcesProvider {

  private static final Logger log = Logger.getLogger(CuVSResourcesProvider.class.getName());
  private static final ThreadLocal<CuVSResources> cuVSResouces;

  static {
    cuVSResouces = ThreadLocal.withInitial(() -> cuVSResourcesOrNull());
  }

  /**
   * Gets an instance of CuVSResources for the accessing thread.
   *
   * @return an instance of CuVSResources
   */
  public static CuVSResources getCuVSResourcesInstance() {
    return cuVSResouces.get();
  }

  /**
   * Sets the instance of CuVSResources
   *
   * @param resources the instance of CuVSResources to set
   */
  public static void set(CuVSResources resources) {
    cuVSResouces.set(resources);
  }

  private static CuVSResources cuVSResourcesOrNull() {
    try {
      return CuVSResources.create();
    } catch (UnsupportedOperationException uoe) {
      log.log(
          Level.WARNING,
          "cuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      if (t instanceof ExceptionInInitializerError ex) {
        t = ex.getCause();
      }
      log.log(Level.WARNING, "Exception occurred during creation of cuVS resources. " + t);
    }
    return null;
  }

  /**
   * Attempts to close the thread's {@link CuVSResources} instance.
   */
  public static void closeCuVSResourcesInstance() {
    CuVSResources r = cuVSResouces.get();
    if (r != null) {
      r.close();
    }
    cuVSResouces.remove();
  }

  /**
   * Checks if cuVS is supported and throws {@link UnsupportedOperationException} otherwise if asked to.
   *
   * @param throwUOE flag that lets this method throw {@link UnsupportedOperationException}
   *
   * @return if cuVS is supported or not
   */
  public static boolean isSupported(boolean throwUOE) {
    boolean isSupported = cuVSResouces.get() != null;
    if (throwUOE && !isSupported) {
      throw new UnsupportedOperationException();
    }
    return isSupported;
  }
}
