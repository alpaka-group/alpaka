#!/bin/bash

#
# Copyright 2014-2015 Benjamin Worpitz
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

# Compiles the project within the directory given by ${1} and executes ${2} within the build folder.

#-------------------------------------------------------------------------------
# e: exit as soon as one command returns a non-zero exit code.
# v: all lines are printed before executing them.
set -ev

#-------------------------------------------------------------------------------
# Build and execute all tests.
oldPath=${PWD}
cd ${1}
mkdir build/
cd build/
mkdir make/
cd make/
cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DBOOST_ROOT="${ALPAKA_BOOST_ROOT_DIR}" -DBOOST_LIBRARYDIR="${ALPAKA_BOOST_LIB_DIR}" -DBoost_COMPILER="${ALPAKA_BOOST_COMPILER}" -DBoost_USE_STATIC_LIBS=ON -DBoost_USE_MULTITHREADED=ON -DBoost_USE_STATIC_RUNTIME=OFF \
    -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE}" -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE}" -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}" -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE="${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" -DALPAKA_ACC_GPU_CUDA_ENABLE="${ALPAKA_ACC_GPU_CUDA_ENABLE}" \
    -DALPAKA_DEBUG="${ALPAKA_DEBUG}" -DALPAKA_INTEGRATION_TEST=ON -DALPAKA_CUDA_VERSION="${ALPAKA_CUDA_VERSION}" -DALPAKA_CUDA_COMPILER="${ALPAKA_CUDA_COMPILER}" \
    "../../"
make VERBOSE=1
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "OFF" ]
then
    eval "${2}"
fi

cd "${oldPath}"
