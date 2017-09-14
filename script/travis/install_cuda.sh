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

# Set the correct CUDA downloads
# NOTE: CUDA < 8.0 did not provide SHA256 in their Release files.
# Installing them in modern Ubuntu versions is therefore not possible.
# We simply add those to the Release files and ignore that they can not be verified during instalation.
ALPAKA_CUDA_CACHE_DIR=${HOME}/cache/CUDA/CUDA-${ALPAKA_CUDA_VER}
if [ "${ALPAKA_CUDA_VER}" == "7.0" ]
then
    ALPAKA_CUDA_PKG_FILE_NAME=cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
    ALPAKA_CUDA_PKG_FILE_PATH=http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/${ALPAKA_CUDA_PKG_FILE_NAME}
elif [ "${ALPAKA_CUDA_VER}" == "7.5" ]
then
    ALPAKA_CUDA_PKG_FILE_NAME=cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
    ALPAKA_CUDA_PKG_FILE_PATH=http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
elif [ "${ALPAKA_CUDA_VER}" == "8.0" ]
then
    ALPAKA_CUDA_PKG_FILE_NAME=cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64-deb
    ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
else
    echo CUDA versions other than 7.0, 7.5 and 8.0 are not currently supported!
fi
if [ -z "$(ls -A "${ALPAKA_CUDA_CACHE_DIR}")" ]
then
    mkdir -p "${ALPAKA_CUDA_CACHE_DIR}"
    travis_retry wget --no-verbose -O "${ALPAKA_CUDA_CACHE_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}" "${ALPAKA_CUDA_PKG_FILE_PATH}"
fi
sudo dpkg --install "${ALPAKA_CUDA_CACHE_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}"
if [ "${ALPAKA_CUDA_VER}" == "7.0" ]
then
    cat /var/cuda-repo-7-0-local/Release
    #cat /var/cuda-repo-7-0-local/Packages.gz | sha256sum
    gunzip -c /var/cuda-repo-7-0-local/Packages.gz | sha256sum
    STR="SHA256:"
    echo "$STR" | sudo tee -a /var/cuda-repo-7-0-local/Release
    cat /var/cuda-repo-7-0-local/Release
elif [ "${ALPAKA_CUDA_VER}" == "7.5" ]
then
    cat /var/cuda-repo-7-5-local/Release
    #cat /var/cuda-repo-7-5-local/Packages.gz | sha256sum
    gunzip -c /var/cuda-repo-7-5-local/Packages.gz | sha256sum
    STR="SHA256:"
    echo "$STR" | sudo tee -a /var/cuda-repo-7-5-local/Release
    cat /var/cuda-repo-7-5-local/Release
fi
travis_retry sudo apt-get -y --quiet update

# Install CUDA
# Currently we do not install CUDA fully: sudo apt-get --quiet -y install cuda
# We only install the minimal packages. Because of our manual partial installation we have to create a symlink at /usr/local/cuda
sudo apt-get -y --quiet --allow-unauthenticated install cuda-core-"${ALPAKA_CUDA_VER}" cuda-cudart-"${ALPAKA_CUDA_VER}" cuda-cudart-dev-"${ALPAKA_CUDA_VER}" cuda-curand-"${ALPAKA_CUDA_VER}" cuda-curand-dev-"${ALPAKA_CUDA_VER}"
sudo ln -s /usr/local/cuda-"${ALPAKA_CUDA_VER}" /usr/local/cuda
export PATH=/usr/local/cuda-${ALPAKA_CUDA_VER}/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${ALPAKA_CUDA_VER}/lib64:$LD_LIBRARY_PATH

if [ "${ALPAKA_CUDA_COMPILER}" == "clang" ]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated install g++-multilib
fi

if [ "${ALPAKA_CUDA_COMPILER}" == "nvcc" ]
then
    which nvcc
    nvcc -V
fi
