#!/bin/bash

#
# Copyright 2021 Antonio Di Pilato
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    brew install libomp
fi

