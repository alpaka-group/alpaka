/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// TODO: delete
// #define ALPAKA_ACC_SYCL_ENABLED 1

#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/sycl/GenericSyclBufImpl.hpp"
#include "alpaka/mem/buf/sycl/MutBufGenericSycl.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <memory>
#include <type_traits>

// TODO: delete
#define ALPAKA_ACC_SYCL_ENABLED 1

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class ConstBufGenericSycl : public internal::ViewAccessOps<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
    public:
        //! Constructor
        template<typename TExtent, typename Deleter>
        ConstBufGenericSycl(DevGenericSycl<TTag> const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
            : m_spBufSyclImpl{std::make_shared<detail::BufSyclImpl<TElem, TDim, TIdx, TTag>>(
                  dev,
                  pMem,
                  std::move(deleter),
                  extent)}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");

            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

        //! Constructor for a ConstBuf from a non-ConstBuf
        ALPAKA_FN_HOST ConstBufGenericSycl(
            MutBufGenericSycl<ConstBufGenericSycl, TElem, DevGenericSycl<TTag>, TElem, TDim, TIdx, TTag> const& buf)
            : m_spBufSyclImpl{buf.m_spBufSyclImpl}
        {
        }

    private:
        std::shared_ptr<detail::BufSyclImpl<TElem, TDim, TIdx, TTag>> m_spBufSyclImpl;

        friend alpaka::trait::GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetExtents<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>;
    };
} // namespace alpaka

#endif
