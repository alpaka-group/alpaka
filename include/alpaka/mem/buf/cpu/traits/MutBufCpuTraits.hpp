/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

namespace alpaka::trait
{
    //! The MutCpuBuf device type trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DevType<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TDev;
    };

    //! The MutCpuBuf device get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetDev<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf) -> TDev
        {
            return GetDev<TBuf<TElem, TDim, TIdx>>::getDev(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutCpuBuf dimension getter trait.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DimType<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The MutCpuBuf memory element type get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct ElemType<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The MutCpuBuf width get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetExtents<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf)
        {
            return GetExtents<TBuf<TElem, TDim, TIdx>>{}(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutCpuBuf native pointer get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetPtrNative<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return GetPtrNative<TBuf<TElem, TDim, TIdx>>::getPtrNative(TBuf<TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>& buf) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrNative<TBuf<TElem, TDim, TIdx>>::getPtrNative(TBuf<TElem, TDim, TIdx>{buf}));
        }
    };

    //! The MutCpuBuf pointer on device get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct GetPtrDev<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>, TDev>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf,
            TDev const& /*dev*/) -> TElem const*
        {
            return GetPtrDev<TBuf<TElem, TDim, TIdx>, TDev>::getPtrDev(TBuf<TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>& buf,
            TDev const& /*dev*/) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrDev<TBuf<TElem, TDim, TIdx>, TDev>::getPtrDev(TBuf<TElem, TDim, TIdx>{buf}));
        }
    };

    //! The MutCpuBuf offset get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct GetOffsets<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf) const
            -> Vec<TDim, TIdx>
        {
            return GetOffsets<TBuf<TElem, TDim, TIdx>>{}(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutCpuBuf idx type trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct IdxType<MutCpuBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait
