#!/bin/bash
#
# Copyright 2023 Antonio Di Pilato, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_omp>"

if [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    brew install libomp
fi
