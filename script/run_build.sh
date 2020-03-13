#!/bin/bash

#
# Copyright 2014-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

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
    if [ ! -z "${1+x}" ] ; then
        echo -n "-D$1=${!1}"
    fi
}

#-------------------------------------------------------------------------------
# Build and execute all tests.
if [ ! -z "${CMAKE_CXX_FLAGS+x}" ]
then
    echo "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
fi
if [ ! -z "${CMAKE_EXE_LINKER_FLAGS+x}" ]
then
    echo "CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
fi
if [ ! -z "${KMP_DEVICE_THREAD_LIMIT+x}" ]
then
    echo "KMP_DEVICE_THREAD_LIMIT=${KMP_DEVICE_THREAD_LIMIT}"
fi
if [ ! -z "${KMP_ALL_THREADS+x}" ]
then
    echo "KMP_ALL_THREADS=${KMP_ALL_THREADS}"
fi
if [ ! -z "${KMP_TEAMS_THREAD_LIMIT+x}" ]
then
    echo "KMP_TEAMS_THREAD_LIMIT=${KMP_TEAMS_THREAD_LIMIT}"
fi
if [ ! -z "${OMP_THREAD_LIMIT+x}" ]
then
    echo "OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT}"
fi
if [ ! -z "${OMP_NUM_THREADS+x}" ]
then
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
fi

mkdir -p build/
cd build/

ALPAKA_CI_CMAKE_GENERATOR_PLATFORM=""
if [ "$TRAVIS_OS_NAME" = "linux" ] || [ "$TRAVIS_OS_NAME" = "osx" ]
then
    ALPAKA_CI_CMAKE_GENERATOR="Unix Makefiles"
elif [ "$TRAVIS_OS_NAME" = "windows" ]
then
    # Use the 64 bit compiler
    # FIXME: Path not found but does not seem to be necessary anymore
    #"./C/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat" amd64

    # Add msbuild to the path
    if [ "$ALPAKA_CI_CL_VER" = "2017" ]
    then
        MSBUILD_EXECUTABLE="/C/Program Files (x86)/Microsoft Visual Studio/2017/Enterprise/MSBuild/15.0/Bin/MSBuild.exe"
    elif [ "$ALPAKA_CI_CL_VER" = "2019" ]
    then
        MSBUILD_EXECUTABLE=$(vswhere.exe -latest -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe")
    fi
    "$MSBUILD_EXECUTABLE" -version

    : ${ALPAKA_CI_CL_VER?"ALPAKA_CI_CL_VER must be specified"}

    # Select the generator
    if [ "$ALPAKA_CI_CL_VER" = "2017" ]
    then
        ALPAKA_CI_CMAKE_GENERATOR="Visual Studio 15 2017"
    elif [ "$ALPAKA_CI_CL_VER" = "2019" ]
    then
        ALPAKA_CI_CMAKE_GENERATOR="Visual Studio 16 2019"
    fi
    ALPAKA_CI_CMAKE_GENERATOR_PLATFORM="-A x64"
fi

cmake -G "${ALPAKA_CI_CMAKE_GENERATOR}" ${ALPAKA_CI_CMAKE_GENERATOR_PLATFORM} \
    "$(env2cmake BOOST_ROOT)" -DBOOST_LIBRARYDIR="${ALPAKA_CI_BOOST_LIB_DIR}/lib" -DBoost_USE_STATIC_LIBS=ON -DBoost_USE_MULTITHREADED=ON -DBoost_USE_STATIC_RUNTIME=OFF \
    "$(env2cmake CMAKE_BUILD_TYPE)" "$(env2cmake CMAKE_CXX_FLAGS)" "$(env2cmake CMAKE_EXE_LINKER_FLAGS)" \
    "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)" \
    "$(env2cmake ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE)" \
    "$(env2cmake ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE)" "$(env2cmake ALPAKA_ACC_CPU_BT_OMP4_ENABLE)" \
    "$(env2cmake TBB_ROOT)" \
    "$(env2cmake ALPAKA_ACC_GPU_CUDA_ENABLE)" "$(env2cmake ALPAKA_CUDA_VERSION)" "$(env2cmake ALPAKA_ACC_GPU_CUDA_ONLY_MODE)" "$(env2cmake ALPAKA_CUDA_ARCH)" "$(env2cmake ALPAKA_CUDA_COMPILER)" \
    "$(env2cmake ALPAKA_CUDA_FAST_MATH)" "$(env2cmake ALPAKA_CUDA_FTZ)" "$(env2cmake ALPAKA_CUDA_SHOW_REGISTER)" "$(env2cmake ALPAKA_CUDA_KEEP_FILES)" "$(env2cmake ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA)" "$(env2cmake ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION)" \
    "$(env2cmake ALPAKA_ACC_GPU_HIP_ENABLE)" "$(env2cmake ALPAKA_ACC_GPU_HIP_ONLY_MODE)" "$(env2cmake ALPAKA_HIP_PLATFORM)" \
    "$(env2cmake ALPAKA_DEBUG)" "$(env2cmake ALPAKA_CI)" "$(env2cmake ALPAKA_CI_ANALYSIS)" "$(env2cmake ALPAKA_CXX_STANDARD)" \
    ".."
if [ "$TRAVIS_OS_NAME" = "linux" ] || [ "$TRAVIS_OS_NAME" = "osx" ]
then
    make VERBOSE=1
elif [ "$TRAVIS_OS_NAME" = "windows" ]
then
    "$MSBUILD_EXECUTABLE" "alpaka.sln" -p:Configuration=${CMAKE_BUILD_TYPE} -maxcpucount:2 -verbosity:minimal
fi

cd ..
