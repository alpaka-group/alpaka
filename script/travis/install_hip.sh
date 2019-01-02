#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
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

source ./script/travis/set.sh

: ${ALPAKA_CI_HIP_ROOT_DIR?"ALPAKA_CI_HIP_ROOT_DIR must be specified"}
: ${ALPAKA_CI_HIP_BRANCH?"ALPAKA_CI_HIP_BRANCH must be specified"}
: ${CMAKE_BUILD_TYPE?"CMAKE_BUILD_TYPE must be specified"}
: ${CXX?"CXX must be specified"}
: ${CC?"CC must be specified"}
: ${ALPAKA_CI_CMAKE_DIR?"ALPAKA_CI_CMAKE_DIR must be specified"}

# CMake
export PATH=${ALPAKA_CI_CMAKE_DIR}/bin:${PATH}
cmake --version

HIP_SOURCE_DIR=${ALPAKA_CI_HIP_ROOT_DIR}/source-hip/
ROCRAND_SOURCE_DIR=${ALPAKA_CI_HIP_ROOT_DIR}/source-rocrand/

git clone -b "${ALPAKA_CI_HIP_BRANCH}" --quiet --recursive --single-branch https://github.com/ROCm-Developer-Tools/HIP.git "${HIP_SOURCE_DIR}"
(cd "${HIP_SOURCE_DIR}"; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX=${ALPAKA_CI_HIP_ROOT_DIR} -DBUILD_TESTING=OFF .. && make && make install)

## rocRAND
# install it into the HIP install dir
git clone --quiet --recursive https://github.com/ROCmSoftwarePlatform/rocRAND "${ROCRAND_SOURCE_DIR}"
(cd "${ROCRAND_SOURCE_DIR}"; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX=${ALPAKA_CI_HIP_ROOT_DIR} -DBUILD_BENCHMARK=OFF -DBUILD_TEST=OFF -DNVGPU_TARGETS="30" -DCMAKE_MODULE_PATH=${ALPAKA_CI_HIP_ROOT_DIR}/cmake .. && make && make install)
