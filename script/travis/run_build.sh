#!/bin/bash

#
# Copyright 2014-2019 Benjamin Worpitz
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

#-------------------------------------------------------------------------------

# create a cmake variable definition if an environment variable exists
#
# This function can not handle environment variables with spaces in its content.
#
# @param $1 cmake/environment variable name
#
# @result if $1 exists cmake variable definition else nothing is returned
#
# @code{.bash}
# FOO=ON
# echo "$(env2cmake FOO)" # returns "-DFOO=ON"
# echo "$(env2cmake BAR)" # returns nothing
# @endcode
function env2cmake()
{
    if [ -v "$1" ] ; then
        echo -n "-D$1=${!1}"
    fi
}

#-------------------------------------------------------------------------------
# Build and execute all tests.
if [[ -v CMAKE_CXX_FLAGS ]]
then
    echo "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
fi
if [[ -v CMAKE_EXE_LINKER_FLAGS ]]
then
    echo "CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
fi
if [[ -v KMP_DEVICE_THREAD_LIMIT ]]
then
    echo "KMP_DEVICE_THREAD_LIMIT=${KMP_DEVICE_THREAD_LIMIT}"
fi
if [[ -v KMP_ALL_THREADS ]]
then
    echo "KMP_ALL_THREADS=${KMP_ALL_THREADS}"
fi
if [[ -v KMP_TEAMS_THREAD_LIMIT ]]
then
    echo "KMP_TEAMS_THREAD_LIMIT=${KMP_TEAMS_THREAD_LIMIT}"
fi
if [[ -v OMP_THREAD_LIMIT ]]
then
    echo "OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT}"
fi
if [[ -v OMP_NUM_THREADS ]]
then
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
fi

mkdir --parents build/
cd build/

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    ALPAKA_CI_CMAKE_GENERATOR="Unix Makefiles"
elif [ "$TRAVIS_OS_NAME" = "windows" ]
then
    # Use the 64 bit compiler
    # FIXME: Path not found but does not seem to be necessary anymore
    #"./C/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat" amd64

    # Add msbuild to the path
    MSBUILD_PATH="/C/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin"
    export PATH=$MSBUILD_PATH:$PATH
    MSBuild.exe -version

    # Select the generator
    ALPAKA_CI_CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
fi

cmake -G "${ALPAKA_CI_CMAKE_GENERATOR}" \
    -DBOOST_ROOT="${ALPAKA_CI_BOOST_ROOT_DIR}" -DBOOST_LIBRARYDIR="${ALPAKA_CI_BOOST_LIB_DIR}/lib" -DBoost_USE_STATIC_LIBS=ON -DBoost_USE_MULTITHREADED=ON -DBoost_USE_STATIC_RUNTIME=OFF \
    "$(env2cmake CMAKE_BUILD_TYPE)" "$(env2cmake CMAKE_CXX_FLAGS)" "$(env2cmake CMAKE_EXE_LINKER_FLAGS)" \
    "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)" \
    "$(env2cmake ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE)" \
    "$(env2cmake ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_BT_OMP4_ENABLE)" \
    "$(env2cmake TBB_ROOT_DIR)" \
    "$(env2cmake ALPAKA_ACC_GPU_CUDA_ENABLE)" "$(env2cmake ALPAKA_CUDA_VERSION)" "$(env2cmake ALPAKA_ACC_GPU_CUDA_ONLY_MODE)" "$(env2cmake ALPAKA_CUDA_ARCH)" "$(env2cmake ALPAKA_CUDA_COMPILER)" \
    "$(env2cmake ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA)" "$(env2cmake ALPAKA_CUDA_NVCC_EXPT_RELAXED_CONSTEXPR)" \
    "$(env2cmake ALPAKA_ACC_GPU_HIP_ENABLE)" "$(env2cmake ALPAKA_ACC_GPU_HIP_ONLY_MODE)" "$(env2cmake ALPAKA_HIP_PLATFORM)" \
    "$(env2cmake ALPAKA_DEBUG)" "$(env2cmake ALPAKA_CI)" "$(env2cmake ALPAKA_CI_ANALYSIS)" \
    ".."
if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    make VERBOSE=1
elif [ "$TRAVIS_OS_NAME" = "windows" ]
then
    MSBuild.exe "alpakaAll.sln" -p:Configuration=${CMAKE_BUILD_TYPE} -maxcpucount:1 -verbosity:minimal
fi

cd ..
