#!/bin/bash

#
# Copyright 2017 Benjamin Worpitz
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

#-------------------------------------------------------------------------------
# e: exit as soon as one command returns a non-zero exit code.
set -e

ALPAKA_CLANG_CACHE_DIR=${HOME}/cache/llvm/llvm-${ALPAKA_CLANG_VER}
if [ -z "$(ls -A "${ALPAKA_CLANG_CACHE_DIR}")" ]
then
    if (( ALPAKA_CLANG_VER_MAJOR >= 5 ))
    then
        ALPAKA_CLANG_PKG_FILE_NAME=clang+llvm-${ALPAKA_CLANG_VER}-linux-x86_64-ubuntu14.04.tar.xz
    else
        ALPAKA_CLANG_PKG_FILE_NAME=clang+llvm-${ALPAKA_CLANG_VER}-x86_64-linux-gnu-ubuntu-14.04.tar.xz
    fi
    travis_retry wget --no-verbose "http://llvm.org/releases/${ALPAKA_CLANG_VER}/${ALPAKA_CLANG_PKG_FILE_NAME}"
    mkdir -p "${ALPAKA_CLANG_CACHE_DIR}"
    xzcat "${ALPAKA_CLANG_PKG_FILE_NAME}" | tar -xf - --strip 1 -C "${ALPAKA_CLANG_CACHE_DIR}"
    sudo rm -rf "${ALPAKA_CLANG_PKG_FILE_NAME}"
fi
"${ALPAKA_CLANG_CACHE_DIR}/bin/llvm-config" --version
export LLVM_CONFIG="${ALPAKA_CLANG_CACHE_DIR}/bin/llvm-config"

# We have to prepend /usr/bin to the path because else the preinstalled clang from usr/bin/local/ is used.
#  travis_retry sudo apt-get -y --quiet --allow-unauthenticated install clang-${ALPAKA_CLANG_VER}

travis_retry sudo apt-get -y --quiet --allow-unauthenticated install libstdc++-"${ALPAKA_CLANG_LIBSTDCPP_VERSION}"-dev
travis_retry sudo apt-get -y --quiet --allow-unauthenticated install libiomp-dev
sudo update-alternatives --install /usr/bin/clang clang "${ALPAKA_CLANG_CACHE_DIR}"/bin/clang 50
sudo update-alternatives --install /usr/bin/clang++ clang++ "${ALPAKA_CLANG_CACHE_DIR}"/bin/clang++ 50
sudo update-alternatives --install /usr/bin/cc cc "${ALPAKA_CLANG_CACHE_DIR}"/bin/clang 50
sudo update-alternatives --install /usr/bin/c++ c++ "${ALPAKA_CLANG_CACHE_DIR}"/bin/clang++ 50
export PATH=${ALPAKA_CLANG_CACHE_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${ALPAKA_CLANG_CACHE_DIR}/lib:${LD_LIBRARY_PATH}
export CPPFLAGS="-I ${ALPAKA_CLANG_CACHE_DIR}/include/c++/v1 ${CPPFLAGS}"
export CXXFLAGS="-lc++ ${CXXFLAGS}"

which "${CXX}"
${CXX} -v
