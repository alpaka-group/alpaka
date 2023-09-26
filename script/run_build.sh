#!/bin/bash
#
# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
# SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
# SPDX-FileContributor: René Widera <r.widera@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
# SPDX-FileContributor: Erik Zenker <erikzenker@posteo.de>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/set.sh

cd build/

if [ -z "${ALPAKA_CI_BUILD_JOBS+x}" ]
then
    ALPAKA_CI_BUILD_JOBS=1
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    make VERBOSE=1 -j${ALPAKA_CI_BUILD_JOBS}
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    "$MSBUILD_EXECUTABLE" "alpaka.sln" -p:Configuration=${CMAKE_BUILD_TYPE} -maxcpucount:${ALPAKA_CI_BUILD_JOBS} -verbosity:minimal
fi

cd ..
