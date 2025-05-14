/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/MutBufCpu.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka
{
    //! The generic memory buffer template implementing muting accessors.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    class MutBufGenericSycl
        : public internal::ViewAccessOps<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST MutBufGenericSycl(TDev const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : m_spBufSyclImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufSyclImpl;
    };
} // namespace alpaka

#endif
