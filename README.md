# Lucene cuVS

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Overview

This library provides a new [KnnVectorFormat](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/KnnVectorsFormat.html) which can be plugged into a Lucene codec.

## Building

### Prerequisites
- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive),
- [Maven 3.9.6+](https://maven.apache.org/download.cgi),
- [JDK 22](https://jdk.java.net/archive/)

```sh
mvn clean compile package
```
The artifacts would be built and available in the target / folder.

### Using with PyLucene

PyLucene embeds a JVM and starts it with the classpath passed to `lucene.initVM(...)`.
Because PyLucene's generated Python module only exposes the Java classes it was built
to wrap, use Lucene's service provider lookup to load `cuvs-lucene` codecs from
Python instead of importing `com.nvidia.cuvs.lucene` classes directly.

Build the PyLucene sidecar jar:

```sh
mvn clean package -DskipTests
```

Then add the generated PyLucene jar to `lucene.CLASSPATH` before starting the VM:

```python
import os
import lucene

cuvs_lucene_jar = "target/cuvs-lucene-26.06.0-jar-with-pylucene-dependencies.jar"
lucene.initVM(classpath=os.pathsep.join([lucene.CLASSPATH, cuvs_lucene_jar]))

from org.apache.lucene.codecs import Codec

codec = Codec.forName("Lucene101AcceleratedHNSWCodec")
```

Use the returned `codec` with `IndexWriterConfig.setCodec(codec)`. The
`jar-with-pylucene-dependencies` artifact includes `cuvs-lucene` and its non-Lucene
runtime dependencies while leaving Lucene itself to PyLucene's own classpath. This
avoids loading a second copy of Lucene classes into the embedded JVM. The regular
`jar-with-dependencies` artifact also merges `META-INF/services` entries and is
available for non-PyLucene Java applications that want a standalone jar.

### Running Tests
```sh
export LD_LIBRARY_PATH={ PATH TO YOUR LOCAL libcuvs_c.so }:$LD_LIBRARY_PATH && mvn clean test
```

## Contributing

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.
