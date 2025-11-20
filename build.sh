#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="26.02.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs"

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg --build-cuvs-java; then
  CUVS_WORKDIR="cuvs-workdir"
  CUVS_GIT_REPO="https://github.com/rapidsai/cuvs.git"
  if [[ -d "$CUVS_WORKDIR" && -n "$(ls -A "$CUVS_WORKDIR")" ]]; then
    echo "Directory '$CUVS_WORKDIR' exists and is not empty."
    pushd $CUVS_WORKDIR
    git pull
  else
    echo "Directory '$CUVS_WORKDIR' does not exist or is empty. Cloning the cuvs repository."
    git clone --branch main $CUVS_GIT_REPO $CUVS_WORKDIR
    pushd $CUVS_WORKDIR
  fi
  if [ -n "${RAPIDS_LOGGER_INCLUDE_DIR:-}" ]; then
    echo "Using user-defined RAPIDS_LOGGER_INCLUDE_DIR: ${RAPIDS_LOGGER_INCLUDE_DIR}"
  elif [ -n "${CONDA_PREFIX:-}" ]; then
    RAPIDS_LOGGER_INCLUDE_DIR="${CONDA_PREFIX}/include"
  else
    echo "Couldn't find a suitable rapids logger include directory."
    exit 1
  fi
  LD_LIBRARY_PATH="$(pwd)/cpp/build/c"
  export LD_LIBRARY_PATH
  ./build.sh libcuvs java
  popd
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
