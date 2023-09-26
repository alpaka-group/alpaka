/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Jakob Krude <jakob.krude@hotmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
#    define ALPAKA_ACC_GPU_CUDA_ENABLED
#endif

#include "alpaka/core/BoostPredef.hpp"

#if defined(BOOST_COMP_CLANG_CUDA) && (BOOST_COMP_CLANG_CUDA == BOOST_VERSION_NUMBER(14, 0, 0))

#    include <cuda.h>

#    if(CUDART_VERSION == 11030)
#        error "clang-14 cannot be used as CUDA compiler when using CUDA v11.3. See alpaka GitHub issue 1857."
#    endif

#endif
