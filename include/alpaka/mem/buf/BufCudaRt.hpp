/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiCudaRt.hpp"
#include "alpaka/mem/Visibility.hpp"
#include "alpaka/mem/buf/BufUniformCudaHipRt.hpp"


#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
    using BufCudaRt = BufUniformCudaHipRt<ApiCudaRt, TElem, TDim, TIdx, TMemVisibility>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
