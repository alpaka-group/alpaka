/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/sycl/BufGenericSycl.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka::trait
{
    //! The SYCL device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct BufType<DevGenericSycl<TTag>, TElem, TDim, TIdx>
    {
        using type = BufGenericSycl<TElem, TDim, TIdx, TTag>;
    };

    //! The BufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DevType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = DevGenericSycl<TTag>;
    };

    //! The BufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetDev<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getDev(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) -> DevGenericSycl<TTag>
        {
            return GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>::getDev(
                ConstBufGenericSycl<TElem, TDim, TIdx, TTag>{buf});
        }
    };

    //! The BufGenericSycl dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DimType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TDim;
    };

    //! The BufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct ElemType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TElem;
    };

    //! The BufGenericSycl width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetExtents<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
        {
            return GetExtents<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>{}(
                ConstBufGenericSycl<TElem, TDim, TIdx, TTag>{buf});
        }
    };

    //! The BufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrNative<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) -> TElem const*
        {
            return GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>::getPtrNative(
                ConstBufGenericSycl<TElem, TDim, TIdx, TTag>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TTag>& buf) -> TElem*
        {
            // cast away the TElem's constness from the ConstBuf's return type
            return const_cast<TElem*>(GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>::getPtrNative(
                ConstBufGenericSycl<TElem, TDim, TIdx, TTag>{buf}));
        }
    };

    //! The BufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrDev<BufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf,
            DevGenericSycl<TTag> const& /*dev*/) -> TElem const*
        {
            return GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>::getPtrDev(
                ConstBufGenericSycl<TElem, TDim, TIdx, TTag>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            BufGenericSycl<TElem, TDim, TIdx, TTag>& buf,
            DevGenericSycl<TTag> const& /*dev*/) -> TElem*
        {
            // cast away the TElem's constness from the ConstBuf's return type
            return const_cast<TElem*>(
                GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>::getPtrDev(
                    ConstBufGenericSycl<TElem, TDim, TIdx, TTag>{buf}));
        }
    };

    //! The BufGenericSycl offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetOffsets<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) const -> Vec<TDim, TIdx>
        {
            return GetOffsets<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>{}(
                ConstBufGenericSycl<TElem, TDim, TIdx, TTag>{buf});
        }
    };

    //! The BufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct IdxType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for Sycl buffers.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct MakeConstBuf<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
            -> ConstBufGenericSycl<TElem, TDim, TIdx, TTag>
        {
            return ConstBufGenericSycl<TElem, TDim, TIdx, TTag>(buf);
        }
    };
} // namespace alpaka::trait

#endif
