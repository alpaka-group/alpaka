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

cd build/

if [ "$TRAVIS_OS_NAME" = "linux" ] || [ "$TRAVIS_OS_NAME" = "osx" ]
then
    make VERBOSE=1
elif [ "$TRAVIS_OS_NAME" = "windows" ]
then
    "$MSBUILD_EXECUTABLE" "alpaka.sln" -p:Configuration=${CMAKE_BUILD_TYPE} -maxcpucount:1 -verbosity:minimal
fi

cd ..
