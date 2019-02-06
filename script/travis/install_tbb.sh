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

source ./script/travis/travis_retry.sh

source ./script/travis/set.sh

# Install TBB
if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libtbb-dev
elif [ "$TRAVIS_OS_NAME" = "osx" ]
then
    brew install tbb
elif [ "$TRAVIS_OS_NAME" = "windows" ]
then
    TBB_ARCHIVE_VER="tbb44_20160526oss"
    TBB_DOWNLOAD_URL="https://www.threadingbuildingblocks.org/sites/default/files/software_releases/windows/${TBB_ARCHIVE_VER}_win_0.zip"
    TBB_DST_PATH="${TBB_ROOT_DIR}/tbb.zip"
    mkdir "${TBB_ROOT_DIR}"
    powershell.exe Invoke-WebRequest "${TBB_DOWNLOAD_URL}" -OutFile "${TBB_DST_PATH}"
    unzip -q "${TBB_DST_PATH}" -d "${TBB_ROOT_DIR}"
    TBB_UNZIP_DIR="${TBB_ROOT_DIR}/${TBB_ARCHIVE_VER}"
    mv ${TBB_UNZIP_DIR}/* "${TBB_ROOT_DIR}/"
    rmdir "${TBB_UNZIP_DIR}"
fi
