# cuVS Lucene

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Contents

1. [What is cuvs-lucene?](#what-is-cuvs-lucene)
2. [Installing cuvs-lucene](#installing-cuvs-lucene)
3. [Getting Started](#getting-started)
4. [Contributing](#contributing)
5. [References](#references)

## What is cuvs-lucene?

`cuvs-lucene` provides a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that uses cuVS to offload vector index build — and optionally search — to NVIDIA GPUs. Because it plugs in through a standard Lucene codec, existing Lucene applications can take advantage of GPU acceleration with minimal code changes and gracefully fall back to the default CPU codec when no GPU is present.

Four codecs are currently provided:

- `Lucene101AcceleratedHNSWCodec` — GPU-accelerated HNSW build with CPU HNSW search. The on-disk format is standard Lucene HNSW, so indexes built on the GPU can be read by any stock Lucene 10.x reader.
  - `LuceneAcceleratedHNSWScalarQuantizedCodec` — scalar-quantized vectors for a smaller index footprint.
  - `LuceneAcceleratedHNSWBinaryQuantizedCodec` — binary-quantized vectors for an even smaller index footprint.
- `CuVS2510GPUSearchCodec` — GPU-accelerated HNSW build and GPU search

## Installing cuvs-lucene

### Prerequisites
- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive)
- [JDK 22](https://jdk.java.net/archive/)
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- The native `libcuvs_c.so` on the runtime library path. Please see the cuVS [Build and Install Guide](https://docs.rapids.ai/api/cuvs/nightly/build/) for install options (conda, pip, tarball, or build from source).

### Maven

To pull `cuvs-lucene` into a Maven project, add the following dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene</artifactId>
  <version>26.06.0</version>
</dependency>
```

### Building from source

```sh
git clone https://github.com/rapidsai/cuvs-lucene.git
cd cuvs-lucene
mvn clean compile package
```

The resulting artifacts are written to `target/`. To run the tests, point `LD_LIBRARY_PATH` at a local `libcuvs_c.so`:

```sh
export LD_LIBRARY_PATH={ PATH TO YOUR LOCAL libcuvs_c.so }:$LD_LIBRARY_PATH && mvn clean test
```

## Getting Started

The snippet below plugs the GPU-accelerated HNSW codec into a standard Lucene `IndexWriter`. Once the codec is set on the `IndexWriterConfig`, indexing proceeds exactly as it would with the default Lucene codec, and search uses the stock `KnnFloatVectorQuery`:

```java
import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
Codec codec = new Lucene101AcceleratedHNSWCodec(params);
IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);

try (Directory dir = FSDirectory.open(indexPath);
    IndexWriter writer = new IndexWriter(dir, config)) {
  Document doc = new Document();
  doc.add(new KnnFloatVectorField("vector_field", embedding, EUCLIDEAN));
  writer.addDocument(doc);
}
```

For fully runnable versions of this example, including one that indexes and searches entirely on the GPU using `CuVS2510GPUSearchCodec`, please refer to the [`examples/`](examples) directory.

## Contributing

If you are interested in contributing to cuvs-lucene, please read our [Contributing guide](CONTRIBUTING.md).

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.

## References

- [Bring Massive-Scale Vector Search to the GPU with Apache Lucene](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71286/) — NVIDIA GTC 2025 session video
- [cuVS and Lucene: GPU-based Vector Search](https://www.youtube.com/watch?v=qiW7iIDFJC0) - Berlin Buzzwords 2024 session video
- [Exploring GPU-accelerated vector search in Elasticsearch with NVIDIA](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia) — Elasticsearch Blog
- [Apache Lucene Accelerated with the NVIDIA cuVS 25.06 Release](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) — SearchScale Blog
