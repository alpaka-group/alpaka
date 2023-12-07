#!/bin/bash
#
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    #-------------------------------------------------------------------------------
    # sloc
    sloccount .

    #-------------------------------------------------------------------------------
    # TODO/FIXME/HACK
    grep -r HACK ./* || true
    grep -r FIXME ./* || true
    grep -r TODO ./* || true

    #-------------------------------------------------------------------------------
    # check shell script with shellcheck
    find . -type f -name "*.sh" -exec shellcheck {} \;
fi
