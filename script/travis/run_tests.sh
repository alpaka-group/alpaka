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

source ./script/travis/set.sh

: ${ALPAKA_ACC_GPU_CUDA_ENABLE?"ALPAKA_ACC_GPU_CUDA_ENABLE must be specified"}
: ${ALPAKA_ACC_GPU_HIP_ENABLE?"ALPAKA_ACC_GPU_HIP_ENABLE must be specified"}

if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "OFF" ] && [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "OFF" ];
then
    cd build/

    if [ "$TRAVIS_OS_NAME" = "linux" ]
    then
        ctest -V
    elif [ "$TRAVIS_OS_NAME" = "windows" ]
    then
        ctest -V -C ${CMAKE_BUILD_TYPE}
    fi

    cd ..
fi
