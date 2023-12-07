# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
# SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

project("alpakaIntegTest" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("axpy/")
add_subdirectory("cudaOnly/")
add_subdirectory("hostOnlyAPI/")
add_subdirectory("mandelbrot/")
add_subdirectory("matMul/")
add_subdirectory("separableCompilation/")
add_subdirectory("sharedMem/")
add_subdirectory("zeroDimBuffer/")
