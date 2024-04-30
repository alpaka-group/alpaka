/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/mem/buf/BufUniformCudaHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufHipRt = BufUniformCudaHipRt<ApiHipRt, TElem, TDim, TIdx>;

    namespace trait
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct MemVisibility<BufHipRt<TElem, TDim, TIdx>>
        {
            using type = std::tuple<alpaka::MemVisibleGpuHipRt>;
        };
    } // namespace trait
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
