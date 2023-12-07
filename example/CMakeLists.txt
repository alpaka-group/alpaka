# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
# SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
# SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
# SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
# SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
# SPDX-FileContributor: Jakob Krude <jakob.krude@hotmail.com>
# SPDX-FileContributor: Jonas Schenke <jonas@groeger-clan.de>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-FileContributor: René Widera <r.widera@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
# SPDX-FileContributor: Erik Zenker <erikzenker@posteo.de>
#
# SPDX-License-Identifier: ISC

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

project("alpakaExamples" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("bufferCopy/")
add_subdirectory("complex/")
add_subdirectory("counterBasedRng/")
add_subdirectory("heatEquation/")
add_subdirectory("helloWorld/")
add_subdirectory("helloWorldLambda/")
add_subdirectory("kernelSpecialization/")
add_subdirectory("ls/")
add_subdirectory("monteCarloIntegration/")
add_subdirectory("openMPSchedule/")
add_subdirectory("parallelLoopPatterns/")
add_subdirectory("randomStrategies/")
add_subdirectory("randomCells2D/")
add_subdirectory("reduce/")
add_subdirectory("tagSpecialization/")
add_subdirectory("vectorAdd/")
add_subdirectory("babelstream/")
