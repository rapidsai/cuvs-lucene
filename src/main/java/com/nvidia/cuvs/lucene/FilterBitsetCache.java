/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.FilterBitsetHandle;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.lucene.search.Query;

/**
 * Shared LRU cache mapping (filter Query, per-segment reader keys, field) → {@link
 * FilterBitsetHandle}.
 *
 * <p>Host-side cache holding packed bitset arrays; the device-side upload is managed inside
 * {@link FilterBitsetHandle} itself (per-thread LRU). Entries are evicted in LRU order and closed
 * on eviction to free the host arrays and signal the device cache.
 */
final class FilterBitsetCache {

  private static final int MAX_HOST_ENTRIES = 16;

  private record FilterCacheKey(Query filter, List<Object> segReaderKeys, String field) {
    @Override
    public boolean equals(Object o) {
      if (!(o instanceof FilterCacheKey other)) return false;
      return Objects.equals(filter, other.filter)
          && Objects.equals(segReaderKeys, other.segReaderKeys)
          && Objects.equals(field, other.field);
    }

    @Override
    public int hashCode() {
      return Objects.hash(filter, segReaderKeys, field);
    }
  }

  private static final LinkedHashMap<FilterCacheKey, FilterBitsetHandle> CACHE =
      new LinkedHashMap<>(MAX_HOST_ENTRIES + 2, 0.75f, /* access-order= */ true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<FilterCacheKey, FilterBitsetHandle> eldest) {
          if (size() > MAX_HOST_ENTRIES) {
            eldest.getValue().close();
            return true;
          }
          return false;
        }
      };

  private FilterBitsetCache() {}

  static synchronized FilterBitsetHandle get(
      Query filter, List<Object> segReaderKeys, String field) {
    return CACHE.get(new FilterCacheKey(filter, segReaderKeys, field));
  }

  static synchronized void put(
      Query filter, List<Object> segReaderKeys, String field, FilterBitsetHandle handle) {
    CACHE.put(new FilterCacheKey(filter, segReaderKeys, field), handle);
  }
}
