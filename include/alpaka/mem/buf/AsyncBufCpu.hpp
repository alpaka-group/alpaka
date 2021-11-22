/* Copyright 2019 Alexander Matthes, Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/core/Vectorize.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <alpaka/core/Cuda.hpp>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/core/Hip.hpp>
#endif

#include <alpaka/mem/alloc/AllocCpuAligned.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <memory>
#include <type_traits>

namespace alpaka
{
    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class AsyncBufCpu
    {
    public:
        template<typename TQueue, typename TExtent>
        ALPAKA_FN_HOST AsyncBufCpu(TQueue queue, TExtent const& extent)
            : m_spBufCpuImpl{
                new detail::BufCpuImpl<TElem, TDim, TIdx>(getDev(queue), extent),
                [queue = std::move(queue)](detail::BufCpuImpl<TElem, TDim, TIdx>* ptr) mutable
                { alpaka::enqueue(queue, [ptr]() { delete ptr; }); }}
        {
            static_assert(
                std::is_same<Dev<TQueue>, DevCpu>::value,
                "The AsyncBufCpu buffer can only be used with a queue on a DevCpu device!");
        }

    public:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufCpuImpl;
    };

    namespace traits
    {
        //! The AsyncBufCpu device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            using type = DevCpu;
        };
        //! The AsyncBufCpu device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetDev<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(AsyncBufCpu<TElem, TDim, TIdx> const& buf) -> DevCpu
            {
                return buf.m_spBufCpuImpl->m_dev;
            }
        };

        //! The AsyncBufCpu dimension getter trait.
        template<typename TElem, typename TDim, typename TIdx>
        struct DimType<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The AsyncBufCpu memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct ElemType<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };
    } // namespace traits
    namespace extent
    {
        namespace traits
        {
            //! The AsyncBufCpu width get trait specialization.
            template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                AsyncBufCpu<TElem, TDim, TIdx>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                ALPAKA_FN_HOST static auto getExtent(AsyncBufCpu<TElem, TDim, TIdx> const& extent) -> TIdx
                {
                    return extent.m_spBufCpuImpl->m_extentElements[TIdxIntegralConst::value];
                }
            };
        } // namespace traits
    } // namespace extent
    namespace traits
    {
        //! The AsyncBufCpu native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(AsyncBufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return buf.m_spBufCpuImpl->m_pMem;
            }
            ALPAKA_FN_HOST static auto getPtrNative(AsyncBufCpu<TElem, TDim, TIdx>& buf) -> TElem*
            {
                return buf.m_spBufCpuImpl->m_pMem;
            }
        };
        //! The AsyncBufCpu pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<AsyncBufCpu<TElem, TDim, TIdx>, DevCpu>
        {
            ALPAKA_FN_HOST static auto getPtrDev(AsyncBufCpu<TElem, TDim, TIdx> const& buf, DevCpu const& dev)
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
            ALPAKA_FN_HOST static auto getPtrDev(AsyncBufCpu<TElem, TDim, TIdx>& buf, DevCpu const& dev) -> TElem*
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
        //! The AsyncBufCpu pitch get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPitchBytes<DimInt<TDim::value - 1u>, AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPitchBytes(AsyncBufCpu<TElem, TDim, TIdx> const& pitch) -> TIdx
            {
                return pitch.m_spBufCpuImpl->m_pitchBytes;
            }
        };

        //! The AsyncBufCpu memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct AsyncBufAlloc<TElem, TDim, TIdx, DevCpu>
        {
            template<typename TQueue, typename TExtent>
            ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent)
                -> AsyncBufCpu<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                return AsyncBufCpu<TElem, TDim, TIdx>(queue, extent);
            }
        };
        //! The AsyncBufCpu memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Map<AsyncBufCpu<TElem, TDim, TIdx>, DevCpu>
        {
            ALPAKA_FN_HOST static auto map(AsyncBufCpu<TElem, TDim, TIdx>& buf, DevCpu const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                {
                    throw std::runtime_error("Memory mapping of AsyncBufCpu between two devices is not implemented!");
                }
                // If it is the same device, nothing has to be mapped.
            }
        };
        //! The AsyncBufCpu memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unmap<AsyncBufCpu<TElem, TDim, TIdx>, DevCpu>
        {
            ALPAKA_FN_HOST static auto unmap(AsyncBufCpu<TElem, TDim, TIdx>& buf, DevCpu const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                {
                    throw std::runtime_error(
                        "Memory unmapping of AsyncBufCpu between two devices is not implemented!");
                }
                // If it is the same device, nothing has to be mapped.
            }
        };
        //! The AsyncBufCpu memory pinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Pin<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto pin(AsyncBufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(!isPinned(buf))
                {
#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                    if(buf.m_spBufCpuImpl->m_extentElements.prod() != 0)
                    {
                        // - cudaHostRegisterDefault:
                        //   See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                        // - cudaHostRegisterPortable:
                        //   The memory returned by this call will be considered as pinned memory by all CUDA contexts,
                        //   not just the one that performed the allocation.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                            ALPAKA_API_PREFIX(HostRegister)(
                                const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                                extent::getExtentProduct(buf) * sizeof(Elem<AsyncBufCpu<TElem, TDim, TIdx>>),
                                ALPAKA_API_PREFIX(HostRegisterDefault)),
                            ALPAKA_API_PREFIX(ErrorHostMemoryAlreadyRegistered));

                        buf.m_spBufCpuImpl->m_bPinned = true;
                    }
#else
                    static_assert(
                        meta::DependentFalseType<TElem>::value,
                        "Memory pinning of AsyncBufCpu is not implemented when CUDA or HIP is not enabled!");
#endif
                }
            }
        };
        //! The AsyncBufCpu memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unpin<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto unpin(AsyncBufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                alpaka::unpin(*buf.m_spBufCpuImpl.get());
            }
        };
        //! The AsyncBufCpu memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IsPinned<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto isPinned(AsyncBufCpu<TElem, TDim, TIdx> const& buf) -> bool
            {
                return alpaka::isPinned(*buf.m_spBufCpuImpl.get());
            }
        };
        //! The AsyncBufCpu memory prepareForAsyncCopy trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct PrepareForAsyncCopy<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto prepareForAsyncCopy(AsyncBufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // to optimize the data transfer performance between a cuda/hip device the cpu buffer has to be pinned,
                // for exclusive cpu use, no preparing is needed
#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                pin(buf);
#else
                alpaka::ignore_unused(buf);
#endif
            }
        };

        //! The AsyncBufCpu offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, AsyncBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getOffset(AsyncBufCpu<TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //! The AsyncBufCpu idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IdxType<AsyncBufCpu<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#include <alpaka/mem/buf/cpu/Copy.hpp>
#include <alpaka/mem/buf/cpu/Set.hpp>
