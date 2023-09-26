# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

project("alpakaAnalysisTest" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("headerCheck/")
