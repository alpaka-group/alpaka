#!/bin/bash
#
# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/travis_retry.sh

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    #-------------------------------------------------------------------------------
    # Install sloc
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install sloccount
    sloccount --version

    #-------------------------------------------------------------------------------
    # Install shellcheck
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install shellcheck
    shellcheck --version

elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    #-------------------------------------------------------------------------------
    # Install sloc
    brew install sloccount
    sloccount --version

    #-------------------------------------------------------------------------------
    # Install shellcheck
    brew install shellcheck
    shellcheck --version

fi
