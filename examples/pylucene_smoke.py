# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import lucene
from lucene import JArray


REPO_ROOT = Path(__file__).resolve().parents[1]
CUVS_LUCENE_JAR = (
    REPO_ROOT / "target" / "cuvs-lucene-26.06.0-jar-with-pylucene-dependencies.jar"
)
INDEX_PATH = Path("/tmp/cuvs-lucene-pylucene-index")


def fvec(values):
    return JArray("float")([float(value) for value in values])


def main():
    lucene.initVM(classpath=os.pathsep.join([lucene.CLASSPATH, str(CUVS_LUCENE_JAR)]))

    from java.nio.file import Paths
    from org.apache.lucene.codecs import Codec
    from org.apache.lucene.document import Document, Field, KnnFloatVectorField, StringField
    from org.apache.lucene.index import (
        DirectoryReader,
        IndexWriter,
        IndexWriterConfig,
        VectorSimilarityFunction,
    )
    from org.apache.lucene.search import IndexSearcher, KnnFloatVectorQuery
    from org.apache.lucene.store import MMapDirectory

    codec = Codec.forName("Lucene101AcceleratedHNSWCodec")

    directory = MMapDirectory(Paths.get(str(INDEX_PATH)))
    config = IndexWriterConfig()
    config.setCodec(codec)
    config.setUseCompoundFile(False)

    writer = IndexWriter(directory, config)
    for doc_id, vector in [("a", [0.1, 0.2, 0.3]), ("b", [0.2, 0.1, 0.4])]:
        doc = Document()
        doc.add(StringField("id", doc_id, Field.Store.YES))
        doc.add(
            KnnFloatVectorField(
                "vector", fvec(vector), VectorSimilarityFunction.EUCLIDEAN
            )
        )
        writer.addDocument(doc)
    writer.close()

    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    query = KnnFloatVectorQuery("vector", fvec([0.1, 0.2, 0.3]), 2)

    for hit in searcher.search(query, 2).scoreDocs:
        print(searcher.doc(hit.doc).get("id"), hit.score)

    reader.close()
    directory.close()


if __name__ == "__main__":
    main()
