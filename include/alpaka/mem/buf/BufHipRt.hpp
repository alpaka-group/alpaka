/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/mem/buf/BufUniformCudaHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
    using BufHipRt = BufUniformCudaHipRt<ApiHipRt, TElem, TDim, TIdx, TMemVisibility>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
