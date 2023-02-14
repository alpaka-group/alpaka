#!/bin/bash

#
# Copyright 2022 Axel Huebl, Simeon Ehrig
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

    travis_retry sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic intel-oneapi-mkl-devel intel-oneapi-openmp intel-oneapi-tbb-devel

    set +eu
    source /opt/intel/oneapi/setvars.sh
    set -eu
fi

which "${CXX}"
${CXX} -v
which "${CC}"
${CC} -v
