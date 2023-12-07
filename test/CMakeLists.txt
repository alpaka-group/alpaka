# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: René Widera <r.widera@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.22)

if(NOT TARGET alpaka::alpaka)
    list(APPEND CMAKE_MODULE_PATH "${alpaka_ROOT}")
    find_package(alpaka REQUIRED)
endif()

add_subdirectory(common)

list(APPEND _alpaka_TEST_OPTIONS --colour-mode default)

add_subdirectory(analysis)
add_subdirectory(integ)
add_subdirectory(unit)
