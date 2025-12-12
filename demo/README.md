# Demo

This maven project contains basic examples that showcase how `cuvs-lucene` can be used.

## Prerequisites
- [Docker](https://www.docker.com/)
- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive)
- A machine with an Nvidia GPU

## Steps

If you are currently in this directory (and to be in the `cuvs-lucene's` root directory) do:
```sh
cd ..
```

Then do:
```sh
docker run --rm --gpus all --pull=always --volume $PWD:$PWD --workdir $PWD -it rapidsai/ci-conda:25.12-cuda13.0.2-ubuntu24.04-py3.13
```

Inside the docker container (and in the `cuvs-lucene's` root directory) do:
```sh
./ci/build_java.sh && conda activate java && cd demo
```

To run Accelerated HNSW demo do:
```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/demo-25.12.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.demo.AcceleratedHnswDemo
```

To run the Index and Search on GPU demo do:
```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/demo-25.12.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.demo.IndexAndSearchonGPUDemo
```
