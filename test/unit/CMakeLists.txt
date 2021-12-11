#
# Copyright 2015-2021 Benjamin Worpitz, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.18)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("mem/buf/")
add_subdirectory("mem/copy/")
add_subdirectory("mem/fence/")
add_subdirectory("mem/view/")
add_subdirectory("mem/p2p/")

