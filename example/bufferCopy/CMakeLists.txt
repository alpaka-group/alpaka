#
# \file
# Copyright 2014-2015 Erik Zenker
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
#

project(alpakaBufferCopy)
cmake_minimum_required(VERSION 3.0.1)

################################################################################
# Find alpaka
################################################################################
set(ALPAKA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" CACHE STRING "The location of the alpaka library")
list(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")

find_package("alpaka" REQUIRED)
include_directories(SYSTEM ${alpaka_INCLUDE_DIRS})
add_definitions(${alpaka_DEFINITIONS})
add_definitions(${ALPAKA_DEV_COMPILE_OPTIONS})
set(LIBS ${LIBS} ${alpaka_LIBRARIES})

###############################################################################
# Executable
###############################################################################
set(bufferCopy "bufferCopy")
file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

alpaka_add_executable(${bufferCopy} ${SRCFILES})
target_link_libraries(${bufferCopy} ${LIBS})

set_target_properties(${bufferCopy} PROPERTIES FOLDER "example")
