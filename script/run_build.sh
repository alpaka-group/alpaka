#!/bin/bash

#
# Copyright 2014-2021 Benjamin Worpitz, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#
set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: run_build>"

cd build/

if [ -z "${ALPAKA_CI_BUILD_JOBS+x}" ]
then
    ALPAKA_CI_BUILD_JOBS=1
fi

# nvcc needs a lot of memory to compile for multiple architectures with optimisations and debug symbols
if [ "${ALPAKA_CI_INSTALL_CUDA:-OFF}" == "ON" ] && [ "${ALPAKA_CI_CUDA_COMPILER:-}" == "nvcc" ] \
    && [ "${CMAKE_BUILD_TYPE:-}" == "RelWithDebInfo" ]
then
    ALPAKA_CI_BUILD_JOBS=$(( (ALPAKA_CI_BUILD_JOBS + 1) / 2 ))
    echo_yellow "nvcc with RelWithDebInfo: reducing the parallel build jobs to ${ALPAKA_CI_BUILD_JOBS}"
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    make VERBOSE=1 -j${ALPAKA_CI_BUILD_JOBS}
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    "$MSBUILD_EXECUTABLE" "alpaka.sln" -p:Configuration=${CMAKE_BUILD_TYPE} -maxcpucount:${ALPAKA_CI_BUILD_JOBS} -verbosity:minimal
fi

cd ..
