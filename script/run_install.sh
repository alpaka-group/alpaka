#!/bin/bash
#
# SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
#
# SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/set.sh

ALPAKA_CI_CMAKE_EXECUTABLE=cmake
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    ALPAKA_CI_CMAKE_EXECUTABLE="${ALPAKA_CI_CMAKE_DIR}/bin/cmake"
fi

"${ALPAKA_CI_CMAKE_EXECUTABLE}" --install build --config ${CMAKE_BUILD_TYPE}
