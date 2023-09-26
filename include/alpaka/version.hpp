/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 * SPDX-FileContributor: Erik Zenker <erikzenker@posteo.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <boost/predef/version_number.h>

#define ALPAKA_VERSION_MAJOR 1
#define ALPAKA_VERSION_MINOR 0
#define ALPAKA_VERSION_PATCH 0

//! The alpaka library version number
#define ALPAKA_VERSION BOOST_VERSION_NUMBER(ALPAKA_VERSION_MAJOR, ALPAKA_VERSION_MINOR, ALPAKA_VERSION_PATCH)
