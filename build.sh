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

function checkIfBranchExists () {
  return "$(eval "git ls-remote --heads $1 refs/heads/$2")"
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
    # Correct branch selection is crucial to avoid version mismatch issues when testing.
    VERSION_IN_FILE=$(cat "VERSION")
    VERSION_SHORT=${VERSION_IN_FILE::-3}
    # First look for a branch example: release/25.12.01 (if exists, when a patch version exists instead of just '00').
    if [[ -n $(checkIfBranchExists $CUVS_GIT_REPO "release/$VERSION_IN_FILE") ]]; then
      echo "The branch: release/$VERSION_IN_FILE exists"
      git clone --branch "release/$VERSION_IN_FILE" $CUVS_GIT_REPO $CUVS_WORKDIR
    # Else look for a branch example: release/25.12.
    elif [[ -n $(checkIfBranchExists $CUVS_GIT_REPO "release/$VERSION_SHORT") ]]; then
      echo "The branch: release/$VERSION_SHORT exists"
      git clone --branch "release/$VERSION_SHORT" $CUVS_GIT_REPO $CUVS_WORKDIR
    # Fallback to the main in the worst case, that is certain to exist.
    else
      echo "Falling back to the main branch for the cuvs repo."
      git clone --branch main $CUVS_GIT_REPO $CUVS_WORKDIR
    fi
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
  ./build.sh java
  popd
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
