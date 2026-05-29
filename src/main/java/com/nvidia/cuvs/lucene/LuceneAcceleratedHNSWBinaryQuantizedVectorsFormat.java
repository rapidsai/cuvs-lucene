/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * cuVS based Binary Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class.getName());
  private static final int MAX_DIMENSIONS = 4096;

  private final AcceleratedHNSWParams acceleratedHNSWParams;

  private static LuceneProvider luceneProvider(String version) {
    try {
      return LuceneProvider.getInstance(version);
    } catch (Exception e) {
      throw new UnsupportedOperationException(
          "Lucene" + version + " vector formats are not available in this runtime", e);
    }
  }

  private static FlatVectorsFormat flatVectorsFormat() {
    try {
      return luceneProvider("102")
          .getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
    } catch (Exception e) {
      throw new UnsupportedOperationException(
          "Binary quantized vectors require Lucene102 vector formats", e);
    }
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat} with default values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   */
  public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(
      AcceleratedHNSWParams acceleratedHNSWParams) {
    super("Lucene99AcceleratedHNSWBinaryQuantizedVectorsFormat");
    this.acceleratedHNSWParams = acceleratedHNSWParams;
  }

  /**
   * Returns a KnnVectorsWriter to write the binary quantized vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    var flatWriter = flatVectorsFormat().fieldsWriter(state);
    if (isSupported()) {
      log.log(
          Level.FINE,
          "cuVS is supported so using the Lucene99AcceleratedHNSWBinaryQuantizedVectorsWriter");
      return new LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter(
          state, acceleratedHNSWParams, flatWriter);
    } else {
      try {
        // Fallback to Lucene's Lucene102HnswBinaryQuantizedVectorsFormat format
        log.log(
            Level.WARNING,
            "GPU based indexing not supported, falling back to using the"
                + " Lucene102HnswBinaryQuantizedVectorsFormat");
        KnnVectorsFormat fallbackFormat =
            luceneProvider("102")
                .getLuceneHnswBinaryQuantizedVectorsFormatInstance(
                    acceleratedHNSWParams.getMaxConn(), acceleratedHNSWParams.getBeamWidth());
        return fallbackFormat.fieldsWriter(state);
      } catch (Exception e) {
        throw new RuntimeException(e.getMessage());
      }
    }
  }

  /**
   * Returns a KnnVectorsReader to read the binary quantized vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    try {
      return luceneProvider("99")
          .getLuceneHnswVectorsReaderInstance(state, flatVectorsFormat().fieldsReader(state));
    } catch (Exception e) {
      throw new RuntimeException(e.getMessage());
    }
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return MAX_DIMENSIONS;
  }
}
