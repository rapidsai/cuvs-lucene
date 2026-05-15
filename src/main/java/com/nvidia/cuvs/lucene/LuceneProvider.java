/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;

/**
 * Dynamically loads Lucene format, reader, and writer classes with a fallback mechanism.
 *
 * @since 25.12
 */
public class LuceneProvider {

  static final Logger log = Logger.getLogger(LuceneProvider.class.getName());

  private static final String BASE = "org.apache.lucene.";
  private static String codecs = "codecs.lucene<version>.";
  private static String fallbackCodecs = "backward_codecs.lucene<version>.";

  private static String luceneFlatVectorsFormat =
      BASE + codecs + "Lucene<version>FlatVectorsFormat";
  private static String luceneFlatVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>FlatVectorsFormat";

  private static String luceneHnswVectorsFormat =
      BASE + codecs + "Lucene<version>HnswVectorsFormat";
  private static String luceneHnswVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswVectorsFormat";

  private static String luceneHnswVectorsReader =
      BASE + codecs + "Lucene<version>HnswVectorsReader";
  private static String luceneHnswVectorsReaderFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswVectorsReader";

  private static String luceneHnswVectorsWriter =
      BASE + codecs + "Lucene<version>HnswVectorsWriter";
  private static String luceneHnswVectorsWriterFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswVectorsWriter";

  private static String luceneBinaryQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>BinaryQuantizedVectorsFormat";
  private static String luceneBinaryQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>BinaryQuantizedVectorsFormat";

  private static String luceneHnswBinaryQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>HnswBinaryQuantizedVectorsFormat";
  private static String luceneHnswBinaryQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswBinaryQuantizedVectorsFormat";

  private static String luceneScalarQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>ScalarQuantizedVectorsFormat";
  private static String luceneScalarQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>ScalarQuantizedVectorsFormat";

  private static String luceneHnswScalarQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>HnswScalarQuantizedVectorsFormat";
  private static String luceneHnswScalarQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswScalarQuantizedVectorsFormat";

  private static String luceneCodec = BASE + codecs + "Lucene<version>Codec";
  private static String luceneCodecFallback = BASE + fallbackCodecs + "Lucene<version>Codec";

  private static final Map<String, LuceneProvider> INSTANCES = new ConcurrentHashMap<>();

  private static MethodHandles.Lookup lookup = MethodHandles.lookup();

  private Class<?> flatVectorsFormat;
  private Class<?> hnswVectorsFormat;
  private Class<?> hnswVectorsReader;
  private Class<?> hnswVectorsWriter;
  private Class<?> binaryQuantizedVectorsFormat;
  private Class<?> hnswBinaryQuantizedVectorsFormat;
  private Class<?> scalarQuantizedVectorsFormat;
  private Class<?> hnswScalarQuantizedVectorsFormat;

  public static LuceneProvider getInstance(String version) throws ClassNotFoundException {
    try {
      return INSTANCES.computeIfAbsent(version, LuceneProvider::newInstanceUnchecked);
    } catch (RuntimeException e) {
      if (e.getCause() instanceof ClassNotFoundException cnfe) {
        throw cnfe;
      }
      throw e;
    }
  }

  private static LuceneProvider newInstanceUnchecked(String version) {
    try {
      return new LuceneProvider(version);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  private LuceneProvider(String version) throws ClassNotFoundException {
    // Lucene 10.4 still stores float HNSW and the default FlatVectorsFormat under lucene99; there
    // is no lucene104 Lucene104FlatVectorsFormat / Lucene104HnswVectorsFormat in lucene-core.
    flatVectorsFormat =
        loadVectorFamilyClass(version, luceneFlatVectorsFormat, luceneFlatVectorsFormatFallback);
    hnswVectorsFormat =
        loadVectorFamilyClass(version, luceneHnswVectorsFormat, luceneHnswVectorsFormatFallback);
    hnswVectorsReader =
        loadVectorFamilyClass(version, luceneHnswVectorsReader, luceneHnswVectorsReaderFallback);
    hnswVectorsWriter =
        loadVectorFamilyClass(version, luceneHnswVectorsWriter, luceneHnswVectorsWriterFallback);
    scalarQuantizedVectorsFormat =
        loadClass(
            setVersion(luceneScalarQuantizedVectorsFormat, version),
            setVersion(luceneScalarQuantizedVectorsFormatFallback, version));

    hnswScalarQuantizedVectorsFormat =
        loadClass(
            setVersion(luceneHnswScalarQuantizedVectorsFormat, version),
            setVersion(luceneHnswScalarQuantizedVectorsFormatFallback, version));

    // Binary-quantized storage lives under lucene102 packages. Lucene 10.4 keeps these types in
    // backward-codecs only; there is no lucene104 BinaryQuantizedVectorsFormat.
    if ("102".equals(version)) {
      binaryQuantizedVectorsFormat =
          loadClass(
              setVersion(luceneBinaryQuantizedVectorsFormat, version),
              setVersion(luceneBinaryQuantizedVectorsFormatFallback, version));
      hnswBinaryQuantizedVectorsFormat =
          loadClass(
              setVersion(luceneHnswBinaryQuantizedVectorsFormat, version),
              setVersion(luceneHnswBinaryQuantizedVectorsFormatFallback, version));
    } else if ("104".equals(version)) {
      final String lucene102BinaryFlat =
          BASE + "backward_codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat";
      final String lucene102HnswBinary =
          BASE + "backward_codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat";
      binaryQuantizedVectorsFormat = loadClass(lucene102BinaryFlat, lucene102BinaryFlat);
      hnswBinaryQuantizedVectorsFormat = loadClass(lucene102HnswBinary, lucene102HnswBinary);
    }
  }

  private static String setVersion(String pkg, String version) {
    return pkg.replaceAll("<version>", version);
  }

  private Class<?> loadVectorFamilyClass(String version, String primaryTpl, String fallbackTpl)
      throws ClassNotFoundException {
    try {
      return loadClass(setVersion(primaryTpl, version), setVersion(fallbackTpl, version));
    } catch (ClassNotFoundException e) {
      if ("99".equals(version)) {
        throw e;
      }
      return loadClass(setVersion(primaryTpl, "99"), setVersion(fallbackTpl, "99"));
    }
  }

  private static Class<?> loadClass(String defaultClassName, String fallbackClassName)
      throws ClassNotFoundException {
    try {
      return Class.forName(defaultClassName);
    } catch (ClassNotFoundException e) {
      // Load class from fallback package.
      try {
        return Class.forName(fallbackClassName);
      } catch (ClassNotFoundException e1) {
        // Should not reach here.
        log.log(Level.SEVERE, "Unable to load class: " + fallbackClassName);
        throw e1;
      }
    }
  }

  public static Codec getCodec(String version)
      throws ClassNotFoundException,
          NoSuchMethodException,
          SecurityException,
          InstantiationException,
          IllegalAccessException,
          IllegalArgumentException,
          InvocationTargetException {
    Class<?> codecClass =
        loadClass(setVersion(luceneCodec, version), setVersion(luceneCodecFallback, version));
    Constructor<?> codecClassConstructor = codecClass.getConstructor();
    return (Codec) codecClassConstructor.newInstance();
  }

  public FlatVectorsFormat getLuceneFlatVectorsFormatInstance(FlatVectorsScorer scorer)
      throws Exception {
    try {
      Constructor<?> luceneFlatVectorsFormatConstructor =
          flatVectorsFormat.getConstructor(FlatVectorsScorer.class);
      return (FlatVectorsFormat) luceneFlatVectorsFormatConstructor.newInstance(scorer);
    } catch (Exception e) {
      log.log(Level.SEVERE, "Unable to initialize LuceneFlatVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  public KnnVectorsReader getLuceneHnswVectorsReaderInstance(
      SegmentReadState state, FlatVectorsReader reader) throws Exception {
    try {
      Constructor<?> luceneHnswVectorsReaderConstructor =
          hnswVectorsReader.getConstructor(SegmentReadState.class, FlatVectorsReader.class);
      return (KnnVectorsReader) luceneHnswVectorsReaderConstructor.newInstance(state, reader);
    } catch (Exception e) {
      log.log(Level.SEVERE, "Unable to initialize LuceneHnswVectorsReader: " + e.getMessage());
      throw e;
    }
  }

  public KnnVectorsWriter getLuceneHnswVectorsWriterInstance(
      SegmentWriteState state,
      int maxConn,
      int beamWidth,
      FlatVectorsWriter writer,
      int numMergeWorkers,
      TaskExecutor executor)
      throws Exception {
    try {
      Constructor<?> luceneHnswVectorsWriterConstructor =
          hnswVectorsWriter.getConstructor(
              SegmentWriteState.class,
              Integer.TYPE,
              Integer.TYPE,
              FlatVectorsWriter.class,
              Integer.TYPE,
              TaskExecutor.class);
      return (KnnVectorsWriter)
          luceneHnswVectorsWriterConstructor.newInstance(
              state, maxConn, beamWidth, writer, numMergeWorkers, executor);
    } catch (Exception e) {
      log.log(Level.SEVERE, "Unable to initialize LuceneHnswVectorsWriter: " + e.getMessage());
      throw e;
    }
  }

  public int getStaticIntParam(String param) throws ReflectiveOperationException {
    try {
      VarHandle varHandle = lookup.findStaticVarHandle(hnswVectorsFormat, param, Integer.TYPE);
      return (int) varHandle.get();
    } catch (NoSuchFieldException | IllegalAccessException e) {
      log.log(Level.SEVERE, "Unable to get " + param + ": " + e.getMessage());
      throw e;
    }
  }

  public List<VectorSimilarityFunction> getSimilarityFunctions()
      throws ReflectiveOperationException {
    try {
      VarHandle varHandle =
          lookup.findStaticVarHandle(hnswVectorsReader, "SIMILARITY_FUNCTIONS", List.class);
      return (List<VectorSimilarityFunction>) varHandle.get();
    } catch (NoSuchFieldException | IllegalAccessException e) {
      log.log(Level.SEVERE, "Unable to get SIMILARITY_FUNCTIONS: " + e.getMessage());
      throw e;
    }
  }

  public FlatVectorsFormat getluceneBinaryQuantizedVectorsFormatInstance() throws Exception {
    try {
      Constructor<?> luceneBinaryQuantizedVectorsFormatConstructor =
          binaryQuantizedVectorsFormat.getConstructor();
      return (FlatVectorsFormat) luceneBinaryQuantizedVectorsFormatConstructor.newInstance();
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneBinaryQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  public KnnVectorsFormat getLuceneHnswBinaryQuantizedVectorsFormatInstance(
      int maxConn, int beamWidth) throws Exception {
    try {
      Constructor<?> ctor =
          hnswBinaryQuantizedVectorsFormat.getConstructor(Integer.TYPE, Integer.TYPE);
      return (KnnVectorsFormat) ctor.newInstance(maxConn, beamWidth);
    } catch (NoSuchMethodException ignored) {
      try {
        Constructor<?> ctor = hnswBinaryQuantizedVectorsFormat.getConstructor();
        return (KnnVectorsFormat) ctor.newInstance();
      } catch (Exception e) {
        log.log(
            Level.SEVERE,
            "Unable to initialize LuceneHnswBinaryQuantizedVectorsFormat: " + e.getMessage());
        throw e;
      }
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneHnswBinaryQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  public FlatVectorsFormat getLuceneScalarQuantizedVectorsFormatInstance() throws Exception {
    try {
      Constructor<?> luceneScalarQuantizedVectorsFormatConstructor =
          scalarQuantizedVectorsFormat.getConstructor();
      return (FlatVectorsFormat) luceneScalarQuantizedVectorsFormatConstructor.newInstance();
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneScalarQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  public KnnVectorsFormat getLuceneHnswScalarQuantizedVectorsFormatInstance(
      int beamWidth, int maxConn) throws Exception {
    try {
      Constructor<?> luceneHnswScalarQuantizedVectorsFormatConstructor =
          hnswScalarQuantizedVectorsFormat.getConstructor(Integer.TYPE, Integer.TYPE);
      return (KnnVectorsFormat)
          luceneHnswScalarQuantizedVectorsFormatConstructor.newInstance(beamWidth, maxConn);
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneHnswScalarQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }
}
