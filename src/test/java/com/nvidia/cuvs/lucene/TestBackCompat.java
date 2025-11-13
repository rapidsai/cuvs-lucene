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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.junit.Test;

/**
 * Tests the backward compatibility mechanism.
 *
 * @since 25.12
 */
public class TestBackCompat {

  @Test
  public void testFallback() throws Exception {
    // Lucene99Codec exists in the org.apache.lucene.backward_codecs.lucene99
    Codec c = LuceneProvider.getCodec("99");
    assertEquals(c.getName(), "Lucene99");
  }

  @Test(expected = ClassNotFoundException.class)
  public void testNonexistentCodec() throws Exception {
    LuceneProvider.getCodec("0");
  }

  @Test
  public void testExistingComponents() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance("99");
    assertTrue(provider.getLuceneFlatVectorsFormatInstance(null) instanceof FlatVectorsFormat);
    assertEquals(provider.getStaticIntParam("VERSION_CURRENT"), 0);
    assertNotEquals(provider.getSimilarityFunctions().size(), 0);
  }
}
