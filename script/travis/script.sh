#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
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

source ./script/travis/set.sh

./script/travis/print_travisEnv.sh
source ./script/travis/before_install.sh

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
  ./script/travis/docker_install.sh
  ./script/travis/docker_run.sh
elif [ "$TRAVIS_OS_NAME" = "windows" ] || [ "$TRAVIS_OS_NAME" = "osx" ]
then
  ./script/travis/install.sh
  ./script/travis/run.sh
fi
