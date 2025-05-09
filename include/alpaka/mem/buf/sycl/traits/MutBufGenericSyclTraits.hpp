/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka::trait
{
    //! The MutBufGenericSycl device type trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct DevType<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        using type = TDev;
    };

    //! The MutBufGenericSycl device get trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct GetDev<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getDev(MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag> const& buf)
            -> TDev
        {
            return GetDev<TBuf<TElem, TDim, TIdx, TTag>>::getDev(TBuf<TElem, TDim, TIdx, TTag>{buf});
        }
    };

    //! The MutBufGenericSycl dimension getter trait.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct DimType<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        using type = TDim;
    };

    //! The MutBufGenericSycl memory element type get trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct ElemType<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        using type = TElem;
    };

    //! The MutBufGenericSycl width get trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct GetExtents<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag> const& buf)
        {
            return GetExtents<TBuf<TElem, TDim, TIdx, TTag>>{}(TBuf<TElem, TDim, TIdx, TTag>{buf});
        }
    };

    //! The MutBufGenericSycl native pointer get trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct GetPtrNative<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(
            MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag> const& buf) -> TElem const*
        {
            return GetPtrNative<TBuf<TElem, TDim, TIdx, TTag>>::getPtrNative(TBuf<TElem, TDim, TIdx, TTag>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>& buf)
            -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrNative<TBuf<TElem, TDim, TIdx, TTag>>::getPtrNative(TBuf<TElem, TDim, TIdx, TTag>{buf}));
        }
    };

    //! The MutBufGenericSycl pointer on device get trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct GetPtrDev<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>, TDev>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag> const& buf,
            TDev const& /*dev*/) -> TElem const*
        {
            return GetPtrDev<TBuf<TElem, TDim, TIdx, TTag>, TDev>::getPtrDev(TBuf<TElem, TDim, TIdx, TTag>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>& buf,
            TDev const& /*dev*/) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrDev<TBuf<TElem, TDim, TIdx, TTag>, TDev>::getPtrDev(TBuf<TElem, TDim, TIdx, TTag>{buf}));
        }
    };

    //! The MutBufGenericSycl offset get trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct GetOffsets<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(
            MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag> const& buf) const -> Vec<TDim, TIdx>
        {
            return GetOffsets<TBuf<TElem, TDim, TIdx, TTag>>{}(TBuf<TElem, TDim, TIdx, TTag>{buf});
        }
    };

    //! The MutBufGenericSycl idx type trait specialization.
    template<
        template<typename, typename, typename, concepts::Tag> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        concepts::Tag TTag>
    struct IdxType<MutBufGenericSycl<TBuf, TBufImpl, TDev, TElem, TDim, TIdx, TTag>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait

#endif
