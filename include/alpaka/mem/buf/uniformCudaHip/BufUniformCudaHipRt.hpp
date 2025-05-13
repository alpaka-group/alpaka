/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include "alpaka/mem/buf/uniformCudaHip/ConstBufUniformCudaHipRt.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/MutBufUniformCudaHipRt.hpp"

namespace alpaka
{
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    using BufUniformCudaHipRt = MutBufUniformCudaHipRt<
        ConstBufUniformCudaHipRt,
        alpaka::detail::BufUniformCudaHipRtImpl<TApi, TElem, TDim, TIdx>,
        TApi,
        DevUniformCudaHipRt<TApi>,
        TElem,
        TDim,
        TIdx>;
} // namespace alpaka

#    include "alpaka/mem/buf/uniformCudaHip/Copy.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/Set.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/traits/ConstBufUniformCudaHipRtTraits.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/traits/MutBufUniformCudaHipRtTraits.hpp"

#endif
