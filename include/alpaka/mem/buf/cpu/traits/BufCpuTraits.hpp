/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"

namespace alpaka::trait
{
    //! The CPU device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct BufType<DevCpu, TElem, TDim, TIdx>
    {
        using type = BufCpu<TElem, TDim, TIdx>;
    };

    //!  The BufCpu device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct DevType<BufCpu<TElem, TDim, TIdx>>
    {
        using type = DevCpu;
    };

    //! The BufCpu device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetDev<BufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(BufCpu<TElem, TDim, TIdx> const& buf) -> DevCpu
        {
            return GetDev<ConstBufCpu<TElem, TDim, TIdx>>::getDev(ConstBufCpu<TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufCpu dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx>
    struct DimType<BufCpu<TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The BufCpu memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct ElemType<BufCpu<TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The BufCpu width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetExtents<BufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufCpu<TElem, TDim, TIdx> const& buf)
        {
            return GetExtents<ConstBufCpu<TElem, TDim, TIdx>>{}(ConstBufCpu<TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufCpu native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrNative<BufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
        {
            return GetPtrNative<ConstBufCpu<TElem, TDim, TIdx>>::getPtrNative(ConstBufCpu<TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx>& buf) -> TElem*
        {
            // cast away the TElem's constness from the ConstBufCpu's return type
            return const_cast<TElem*>(
                GetPtrNative<ConstBufCpu<TElem, TDim, TIdx>>::getPtrNative(ConstBufCpu<TElem, TDim, TIdx>{buf}));
        }
    };

    //! The BufCpu pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevCpu>
    {
        ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevCpu const& /*dev*/)
            -> TElem const*
        {
            return GetPtrDev<ConstBufCpu<TElem, TDim, TIdx>, DevCpu>::getPtrDev(ConstBufCpu<TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevCpu const& /*dev*/) -> TElem*
        {
            // cast away the TElem's constness from the ConstBufCpu's return type
            return const_cast<TElem*>(
                GetPtrDev<ConstBufCpu<TElem, TDim, TIdx>, DevCpu>::getPtrDev(ConstBufCpu<TElem, TDim, TIdx>{buf}));
        }
    };

    //! The BufCpu offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetOffsets<BufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufCpu<TElem, TDim, TIdx> const& buf) const -> Vec<TDim, TIdx>
        {
            return GetOffsets<ConstBufCpu<TElem, TDim, TIdx>>{}(ConstBufCpu<TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufCpu idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct IdxType<BufCpu<TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for CPU buffers.
    template<typename TElem, typename TDim, typename TIdx>
    struct MakeConstBuf<BufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(BufCpu<TElem, TDim, TIdx> const& buf) -> ConstBufCpu<TElem, TDim, TIdx>
        {
            return ConstBufCpu<TElem, TDim, TIdx>(buf);
        }
    };
} // namespace alpaka::trait
