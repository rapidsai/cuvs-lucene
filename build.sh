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

if [ -z "${LD_LIBRARY_PATH:-}" ]; then
    echo "==> LD_LIBRARY_PATH not set or is empty so cloning cuvs repository and building it"
    CUVS_REPO="git@github.com:rapidsai/cuvs.git"
    CUVS_DIR="cuvs"
    BUILD_DIR=$CUVS_DIR/cpp/build
    LIBCUVS_SO_FILE=$BUILD_DIR/libcuvs_c.so
    CUR_DIR=$(pwd)
    if [ -e "$LIBCUVS_SO_FILE" ]; then
        export LD_LIBRARY_PATH="$CUR_DIR/$BUILD_DIR"
        echo "==> LD_LIBRARY_PATH is set to: $LD_LIBRARY_PATH (skipping build step as so file is already present)"
    else
        echo "==> Current working directory is: $CUR_DIR, checking out cuvs repository into $CUR_DIR..."
        # checkout cuvs
        RV=$(git clone $CUVS_REPO $CUVS_DIR)
        if [ "$RV" -ne 0 ]; then
            echo "==> cuvs checkout failed."
            exit "$RV"
        else
            # build cuvs
            echo "==> cuvs checkout succeeded, building libcuvs"
            if ! ./$CUVS_DIR/build.sh libcuvs; then
                echo "==> cuvs build.sh returned non-zero."
                exit "$RV"
            fi
            if [ -e "$LIBCUVS_SO_FILE" ]; then
                export LD_LIBRARY_PATH="$CUR_DIR/$BUILD_DIR"
                echo "==> LD_LIBRARY_PATH is set to: $LD_LIBRARY_PATH"
            else
                echo "==> Not setting LD_LIBRARY_PATH as libcuvs_c.so not found."
                exit 1
            fi
        fi
    fi
else
    echo "==> LD_LIBRARY_PATH is set and non-empty, skipping the cuvs build step."
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
