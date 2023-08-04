#!/bin/bash
#
# Copyright 2023 Axel Hübl, Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${CXX?'CXX must be specified'}"


if ! agc-manager -e oneapi
then
    # Ref.: https://github.com/rscohn2/oneapi-ci
    # intel-basekit intel-hpckit are too large in size

    travis_retry sudo apt-get -qqq update
    travis_retry sudo apt-get install -y wget build-essential pkg-config cmake ca-certificates gnupg
    travis_retry sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB

    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
    echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

    travis_retry sudo apt-get update

    #  See a list of oneAPI packages available for install
    echo "################################"
    sudo -E apt-cache pkgnames intel
    echo "################################"

    # Intel mixes different version numbers in releases. Make sure we install the versions that are compatible.
    if [ "${ALPAKA_CI_ONEAPI_VERSION}" == "2023.1.0" ]
    then
        ALPAKA_CI_TBB_VERSION="2021.9.0"
    elif [ "${ALPAKA_CI_ONEAPI_VERSION}" == "2023.2.0" ]
    then
        ALPAKA_CI_TBB_VERSION="2021.10.0"
    fi

    components=(
        intel-oneapi-common-vars                                      # Contains /opt/intel/oneapi/setvars.sh - has no version number
        intel-oneapi-compiler-dpcpp-cpp-"${ALPAKA_CI_ONEAPI_VERSION}" # Contains icpx compiler and SYCL runtime
        intel-oneapi-openmp-"${ALPAKA_CI_ONEAPI_VERSION}"             # For OpenMP back-ends
        intel-oneapi-runtime-opencl                                   # Required to run SYCL tests on the CPU - has no version number
        intel-oneapi-tbb-devel-"${ALPAKA_CI_TBB_VERSION}"             # For TBB back-end
    )
    travis_retry sudo apt-get install -y "${components[@]}"

    set +eu
    source /opt/intel/oneapi/setvars.sh
    set -eu
fi

which "${CXX}"
${CXX} -v
which "${CC}"
${CC} -v
sycl-ls
