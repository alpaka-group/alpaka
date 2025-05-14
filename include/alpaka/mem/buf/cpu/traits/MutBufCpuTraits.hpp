/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/MutBufCpu.hpp"

namespace alpaka::trait
{
    //! The MutBufCpu device type trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DevType<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TDev;
    };

    //! The MutBufCpu device get trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetDev<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf) -> TDev
        {
            return GetDev<TBuf<TElem, TDim, TIdx>>::getDev(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutBufCpu dimension getter trait.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DimType<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The MutBufCpu memory element type get trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct ElemType<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The MutBufCpu width get trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetExtents<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf)
        {
            return GetExtents<TBuf<TElem, TDim, TIdx>>{}(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutBufCpu native pointer get trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetPtrNative<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return GetPtrNative<TBuf<TElem, TDim, TIdx>>::getPtrNative(TBuf<TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>& buf) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrNative<TBuf<TElem, TDim, TIdx>>::getPtrNative(TBuf<TElem, TDim, TIdx>{buf}));
        }
    };

    //! The MutBufCpu pointer on device get trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct GetPtrDev<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>, TDev>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf,
            TDev const& /*dev*/) -> TElem const*
        {
            return GetPtrDev<TBuf<TElem, TDim, TIdx>, TDev>::getPtrDev(TBuf<TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>& buf,
            TDev const& /*dev*/) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrDev<TBuf<TElem, TDim, TIdx>, TDev>::getPtrDev(TBuf<TElem, TDim, TIdx>{buf}));
        }
    };

    //! The MutBufCpu offset get trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct GetOffsets<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf) const
            -> Vec<TDim, TIdx>
        {
            return GetOffsets<TBuf<TElem, TDim, TIdx>>{}(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutBufCpu idx type trait specialization.
    template<
        template<typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct IdxType<MutBufCpu<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait
