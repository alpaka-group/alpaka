/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/alloc/AllocCpuAligned.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/platform/PlatformCpu.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The constant CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class ConstBufCpu : public internal::ViewAccessOps<ConstBufCpu<TElem, TDim, TIdx>>
    {
    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST ConstBufCpu(DevCpu const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : m_spBufCpuImpl{
                  std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
        {
        }

        ALPAKA_FN_HOST ConstBufCpu(BufCpu<TElem, TDim, TIdx> const& other) : m_spBufCpuImpl{other.m_spBufCpuImpl}
        {
        }

    public:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufCpuImpl;
    };

    namespace trait
    {
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
                return buf.m_spBufCpuImpl->m_dev;
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
            using type = TElem;
        };

        //! The ConstBufCpu width get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetExtents<ConstBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST auto operator()(ConstBufCpu<TElem, TDim, TIdx> const& buf)
            {
                return buf.m_spBufCpuImpl->m_extentElements;
            }
        };

        //! The ConstBufCpu native pointer get trait specialization. Only returns constant pointers.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<ConstBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(ConstBufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return buf.m_spBufCpuImpl->m_pMem;
            }
        };

        //! The ConstBufCpu pointer on device get trait specialization. Only returns constant pointers.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<ConstBufCpu<TElem, TDim, TIdx>, DevCpu>
        {
            ALPAKA_FN_HOST static auto getPtrDev(ConstBufCpu<TElem, TDim, TIdx> const& buf, DevCpu const& dev)
                -> TElem const*
            {
                if(dev == getDev(buf))
                {
                    return buf.m_spBufCpuImpl->m_pMem;
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
    } // namespace trait
} // namespace alpaka
