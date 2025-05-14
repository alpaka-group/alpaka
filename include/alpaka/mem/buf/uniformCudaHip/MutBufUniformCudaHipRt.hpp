/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/uniformCudaHip/BufUniformCudaHipRtImpl.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{

    //! The generic memory buffer template implementing muting accessors.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    class MutBufUniformCudaHipRt
        : public internal::ViewAccessOps<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST MutBufUniformCudaHipRt(
            TDev const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extent,
            std::size_t pitchBytes)
            : m_spBufImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent, pitchBytes)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufImpl;
    };

} // namespace alpaka

#endif
