#!/bin/bash
#
# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/travis_retry.sh

source ./script/set.sh

#-------------------------------------------------------------------------------
if [ "$alpaka_CI" = "GITHUB" ]
then
    echo GITHUB_WORKSPACE: "${GITHUB_WORKSPACE}"
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    # Show all running services
    sudo service --status-all

    # Show memory stats
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install smem
    sudo smem
    sudo free -m -t
fi
