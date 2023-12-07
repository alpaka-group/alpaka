#!/bin/bash
#
# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
#
# SPDX-FileContributor: Antonio Di Pilato <tony.dipilato03@gmail.com>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/travis_retry.sh

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    brew reinstall --build-from-source --formula ./script/homebrew/${ALPAKA_CI_XCODE_VER}/libomp.rb
fi
