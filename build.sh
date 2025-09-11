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

function setup_cuvs_from_nightly {
    echo "Trying to pull from a nightly"
    if [ ! -d "libcuvs-env" ]; then
        python3 -m venv libcuvs-env
    fi
    # shellcheck disable=SC1091
    source libcuvs-env/bin/activate
    echo "Installing libcuvs-cu13>=$VERSION via pip..."
    NEXT_MINOR_VERSION=$(echo "$VERSION" | awk -F. '{if($2>12) print $1+1".1"; else print $1"."$2+1}')
    pip install libcuvs-cu13\<"$NEXT_MINOR_VERSION" --pre --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/
    echo "Done pip install!"
    SITE_PACKAGES_PATH=$(find libcuvs-env -name site-packages)
    export VENV_LIB=$SITE_PACKAGES_PATH/libcuvs/lib64:$SITE_PACKAGES_PATH/librmm/lib64:$SITE_PACKAGES_PATH/rapids_logger/lib64
    if [ -z ${LD_LIBRARY_PATH+x} ]
       then export LD_LIBRARY_PATH=/usr/local/cuda-13/targets/x86_64-linux/lib:$VENV_LIB
       else export LD_LIBRARY_PATH=/usr/local/cuda-13/targets/x86_64-linux/lib:$VENV_LIB:${LD_LIBRARY_PATH}
    fi
    deactivate
}

# Set LD_LIBRARY_PATH if not already set
if [ -z "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH=""
fi

# Verify libcuvs_c.so is available
echo "Checking for libcuvs_c.so..."
# Check if LD_LIBRARY_PATH is not empty and not just spaces
if [ -n "$LD_LIBRARY_PATH" ] && [ -n "$(echo "$LD_LIBRARY_PATH" | tr -d '[:space:]')" ]; then
    FOUND_LIB=$(find "${LD_LIBRARY_PATH//:/ }" -maxdepth 1 -name "libcuvs_c.so" 2>/dev/null | head -1)
    if [ -n "$FOUND_LIB" ]; then
        echo "libcuvs_c.so was found in $(dirname "$FOUND_LIB")"
    else
        echo "libcuvs_c.so was not found, LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
        setup_cuvs_from_nightly
    fi
else
    echo "LD_LIBRARY_PATH is not set or empty. libcuvs_c.so may not be found."
    setup_cuvs_from_nightly
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
