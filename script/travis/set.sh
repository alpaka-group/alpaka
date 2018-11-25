#!/bin/bash

#
# Copyright 2018 Benjamin Worpitz
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

#-------------------------------------------------------------------------------
# -e: exit as soon as one command returns a non-zero exit code
# -o pipefail: pipeline returns exit code of the rightmost command with a non-zero exit code
# -u: treat unset variables as an error
# -v: Print shell input lines as they are read
# -x: Print command traces before executing command
set -eouvx pipefail
