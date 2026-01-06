/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSResources;
import java.util.logging.Level;
import java.util.logging.Logger;

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
  public static CuVSResources getInstance() {
    return cuVSResouces.get();
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
   * Tells whether the platform supports cuVS.
   *
   * @return if cuVS is supported or not
   */
  public static boolean supported() {
    return cuVSResouces.get() != null;
  }

  /**
   * Checks if cuVS is supported and throws {@link UnsupportedOperationException} otherwise.
   */
  public static void checkSupported() {
    if (!supported()) {
      throw new UnsupportedOperationException();
    }
  }

  /**
   * Sets the instance of CuVSResources
   *
   * @param resources the instance of CuVSResources to set
   */
  public static void setResources(CuVSResources resources) {
    cuVSResouces.set(resources);
  }
}
