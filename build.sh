#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="25.10.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs"

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg --build-cuvs-from-source; then
    unset LD_LIBRARY_PATH
    CUVS_REPO="git@github.com:rapidsai/cuvs.git"
    CUVS_DIR="cuvs"
    BUILD_DIR=$CUVS_DIR/cpp/build
    CUR_DIR=$(pwd)

    # checkout cuvs
    if [ ! -d "$CUVS_DIR" ]; then
        echo "==> Current working directory is: $CUR_DIR, checking out cuvs repository into $CUR_DIR..."
        RV=$(git clone $CUVS_REPO $CUVS_DIR)
        if [ "$RV" -ne 0 ]; then
            echo "==> cuvs checkout failed."
            exit "$RV"
        fi
    fi

    # build cuvs
    echo "==> Building cuvs"
    if ! ./$CUVS_DIR/build.sh libcuvs; then
        echo "==> cuvs build.sh returned non-zero."
        exit "$RV"
    fi

    # set LD_LIBRARY_PATH variable
    export LD_LIBRARY_PATH="$CUR_DIR/$BUILD_DIR"
    echo "==> LD_LIBRARY_PATH is set to: $LD_LIBRARY_PATH"
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
