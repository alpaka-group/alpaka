#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

source ./script/travis/travis_retry.sh

source ./script/travis/set.sh

: ${ALPAKA_CI_ANALYSIS?"ALPAKA_CI_ANALYSIS must be specified"}
: ${ALPAKA_CI_INSTALL_CUDA?"ALPAKA_CI_INSTALL_CUDA must be specified"}
: ${ALPAKA_CI_INSTALL_HIP?"ALPAKA_CI_INSTALL_HIP must be specified"}
: ${ALPAKA_CI_INSTALL_TBB?"ALPAKA_CI_INSTALL_TBB must be specified"}

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    travis_retry apt-get -y --quiet update
    travis_retry apt-get -y install sudo

    # software-properties-common: 'add-apt-repository' and certificates for wget https download
    # binutils: ld
    # xz-utils: xzcat
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install software-properties-common wget git make binutils xz-utils

    ./script/travis/install_cmake.sh
    if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then ./script/travis/install_analysis.sh ;fi
    # Install CUDA before installing gcc as it installs gcc-4.8 and overwrites our selected compiler
    if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ] ;then ./script/travis/install_cuda.sh ;fi
    if [ "${CXX}" == "g++" ] ;then ./script/travis/install_gcc.sh ;fi
    if [ "${CXX}" == "clang++" ] ;then source ./script/travis/install_clang.sh ;fi
    if [ "${ALPAKA_CI_INSTALL_HIP}" == "ON" ] ;then ./script/travis/install_hip.sh ;fi
fi

if [ "${ALPAKA_CI_INSTALL_TBB}" = "ON" ]
then
    ./script/travis/install_tbb.sh
fi

./script/travis/install_boost.sh

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    # Minimize docker image size
    sudo apt-get --quiet --purge autoremove
    sudo apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
fi
