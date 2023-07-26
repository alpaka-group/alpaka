/* Copyright 2022 Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <cstdio>

// TODO: SYCL doesn't have a way to detect if we're looking at device or host code. This needs a workaround so that
// SYCL and other back-ends are compatible.
#ifdef ALPAKA_ACC_SYCL_ENABLED
#    define ALPAKA_CHECK_DO(printMsg, file, line, success, expression)                                                \
        do                                                                                                            \
        {                                                                                                             \
            if(!(expression))                                                                                         \
            {                                                                                                         \
                acc.cout << "ALPAKA_CHECK failed because '!(" << #expression << ")' in " << file << ":" << line       \
                         << "\n";                                                                                     \
                success = false;                                                                                      \
            }                                                                                                         \
        } while(0)
#else
#    define ALPAKA_CHECK_DO(printMsg, file, line, success, expression)                                                \
        do                                                                                                            \
        {                                                                                                             \
            if(!(expression))                                                                                         \
            {                                                                                                         \
                printf("ALPAKA_CHECK failed because '!(%s)' in %s:%d\n", #expression, file, line);                    \
                success = false;                                                                                      \
            }                                                                                                         \
        } while(0)
#endif

#if BOOST_LANG_HIP == BOOST_VERSION_NUMBER(4, 5, 0)
// disable message print to avoid compiler error: 'error: stack size limit exceeded (181896) in _ZN6alpaka...'
#    define ALPAKA_CHECK(printMsg, file, line, success, expression)                                                   \
        do                                                                                                            \
        {                                                                                                             \
            if(!(expression))                                                                                         \
            {                                                                                                         \
                printf("ALPAKA_CHECK failed because '!(%s)' in %s:%d\n", #expression, file, line);                    \
                success = false;                                                                                      \
            }                                                                                                         \
        } while(0)
#else
#    define ALPAKA_CHECK(success, expression) ALPAKA_CHECK_DO(__FILE__, __LINE__, success, expression)
#endif
