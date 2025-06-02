/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber,
 * Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"
#include "alpaka/mem/buf/cpu/ConstBufCpu.hpp"

namespace alpaka::trait
{
    //! The CPU device memory const-buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct ConstBufType<DevCpu, TElem, TDim, TIdx>
    {
        using type = ConstBufCpu<TElem, TDim, TIdx>;
    };

    //! The ConstBufCpu device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct DevType<ConstBufCpu<TElem, TDim, TIdx>>
    {
        using type = DevCpu;
    };

    //! The ConstBufCpu device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetDev<ConstBufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(ConstBufCpu<TElem, TDim, TIdx> const& buf) -> DevCpu
        {
            return buf.m_spBufImpl->m_dev;
        }
    };

    //! The ConstBufCpu dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx>
    struct DimType<ConstBufCpu<TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The ConstBufCpu memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct ElemType<ConstBufCpu<TElem, TDim, TIdx>>
    {
        // const qualify the element type of the inner view
        using type = TElem const;
    };

    //! The ConstBufCpu width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetExtents<ConstBufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufCpu<TElem, TDim, TIdx> const& buf)
        {
            return buf.m_spBufImpl->m_extentElements;
        }
    };

    //! The ConstBufCpu native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrNative<ConstBufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(ConstBufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
        {
            return buf.m_spBufImpl->m_pMem;
        }
    };

    //! The ConstBufCpu pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<ConstBufCpu<TElem, TDim, TIdx>, DevCpu>
    {
        ALPAKA_FN_HOST static auto getPtrDev(ConstBufCpu<TElem, TDim, TIdx> const& buf, DevCpu const& dev)
            -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spBufImpl->m_pMem;
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }
    };

    //! The ConstBufCpu offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetOffsets<ConstBufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufCpu<TElem, TDim, TIdx> const&) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The ConstBufCpu idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct IdxType<ConstBufCpu<TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for constant CPU buffers.
    template<typename TElem, typename TDim, typename TIdx>
    struct MakeConstBuf<ConstBufCpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufCpu<TElem, TDim, TIdx> const& buf)
            -> ConstBufCpu<TElem, TDim, TIdx>
        {
            return buf;
        }

        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufCpu<TElem, TDim, TIdx>&& buf) -> ConstBufCpu<TElem, TDim, TIdx>
        {
            return buf;
        }
    };
} // namespace alpaka::trait
