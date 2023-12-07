/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Luca Ferragina <luca.ferragina@cern.ch>
 * SPDX-FileContributor: Aurora Perego <aurora.perego@cern.ch>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Sycl.hpp"

#include <cstdio>

#define ALPAKA_CHECK(success, expression)                                                                             \
    do                                                                                                                \
    {                                                                                                                 \
        if(!(expression))                                                                                             \
        {                                                                                                             \
            printf("ALPAKA_CHECK failed because '!(%s)'\n", #expression);                                             \
            success = false;                                                                                          \
        }                                                                                                             \
    } while(0)
