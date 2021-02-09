/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cstdio>

#ifndef ALPAKA_ACC_SYCL_ENABLED
#   define ALPAKA_CHECK(success, expression)                                                                          \
    do                                                                                                                \
    {                                                                                                                 \
        if(!(expression))                                                                                             \
        {                                                                                                             \
            printf("ALPAKA_CHECK failed because '!(%s)'\n", #expression);                                             \
            success = false;                                                                                          \
        }                                                                                                             \
    } while ( 0 )
#else
#   define ALPAKA_CHECK(success, expression)                                                                          \
    do                                                                                                                \
    {                                                                                                                 \
        if(!(expression))                                                                                             \
        {                                                                                                             \
            acc.stream << "ALPAKA_CHECK failed because '!(" << #expression << ")'\n";                                 \
            success = false;                                                                                          \
        }                                                                                                             \
    } while ( 0 )
#endif
