# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
# SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
# SPDX-FileContributor: Antonio Di Pilato <tony.dipilato03@gmail.com>
# SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
# SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
# SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
# SPDX-FileContributor: Jakob Krude <jakob.krude@hotmail.com>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: René Widera <r.widera@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
# SPDX-FileContributor: Erik Zenker <erikzenker@posteo.de>
#
# SPDX-License-Identifier: MPL-2.0

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("acc/")
add_subdirectory("atomic/")
add_subdirectory("block/shared/")
add_subdirectory("block/sharedSharing/")
add_subdirectory("block/sync/")
add_subdirectory("core/")
add_subdirectory("dev/")
add_subdirectory("event/")
add_subdirectory("idx/")
add_subdirectory("intrinsic/")
add_subdirectory("kernel/")
add_subdirectory("math/")
add_subdirectory("mem/buf/")
add_subdirectory("mem/copy/")
add_subdirectory("mem/fence/")
add_subdirectory("mem/p2p/")
add_subdirectory("mem/view/")
add_subdirectory("meta/")
add_subdirectory("queue/")
add_subdirectory("rand/")
add_subdirectory("traits/")
add_subdirectory("vec/")
add_subdirectory("warp/")
add_subdirectory("workDiv/")
