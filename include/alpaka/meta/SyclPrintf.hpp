/* Copyright 2022 Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <CL/sycl.hpp>

#ifdef ALPAKA_ACC_SYCL_ENABLED

// Kill printf in AMD GPU code because of missing compiler support
#    ifdef __AMDGCN__
#        include <cstdio> // the define breaks <cstdio> if it is included afterwards
#        define printf(...)
#    else

#        ifdef __SYCL_DEVICE_ONLY__
#            define CONSTANT __attribute__((opencl_constant))
#        else
#            define CONSTANT
#        endif

#        define printf(FORMAT, ...)                                                                                   \
            do                                                                                                        \
            {                                                                                                         \
                static const CONSTANT char format[] = FORMAT;                                                         \
                sycl::ext::oneapi::experimental::printf(format, ##__VA_ARGS__);                                       \
            } while(false)
#    endif
#endif
