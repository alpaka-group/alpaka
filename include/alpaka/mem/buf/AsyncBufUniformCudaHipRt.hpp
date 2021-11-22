/* Copyright 2019 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        error HIP does not support stream-ordered memory allocations
#    endif

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <functional>
#    include <memory>
#    include <type_traits>

namespace alpaka
{
    class DevUniformCudaHipRt;

    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    //! The CUDA/HIP stream-ordered memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class AsyncBufUniformCudaHipRt
    {
        static_assert(
            !std::is_const<TElem>::value,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const<TIdx>::value, "The idx type of the buffer can not be const!");

    private:
        using Elem = TElem;
        using Dim = TDim;

    public:
        //! Constructor
        template<typename TQueue, typename TExtent>
        ALPAKA_FN_HOST AsyncBufUniformCudaHipRt(
            TQueue queue,
            TElem* const pMem,
            TIdx const& pitchBytes,
            TExtent const& extent)
            : m_dev(getDev(queue))
            , m_extentElements(extent::getExtentVecEnd<TDim>(extent))
            , m_spMem(pMem, [queue](TElem* const ptr) { AsyncBufUniformCudaHipRt::freeBuffer(ptr, queue); })
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == alpaka::Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");
            static_assert(
                std::is_same<TIdx, Idx<TExtent>>::value,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

    private:
        //! Frees the shared buffer.
        template<typename TQueue>
        ALPAKA_FN_HOST static auto freeBuffer(TElem* const memPtr, TQueue const& queue) -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // FIXME Do we really need to set the current device ?
            // Set the current device.
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(getDev(queue).m_iDevice));
            // Free the buffer.
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(
                FreeAsync)(reinterpret_cast<void*>(memPtr), queue.m_spQueueImpl->m_UniformCudaHipQueue));
        }

    public:
        DevUniformCudaHipRt m_dev;
        Vec<TDim, TIdx> m_extentElements;
        std::shared_ptr<TElem> m_spMem;
    };

    namespace traits
    {
        //! The AsyncBufUniformCudaHipRt device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = DevUniformCudaHipRt;
        };
        //! The AsyncBufUniformCudaHipRt device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetDev<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx> const& buf)
                -> DevUniformCudaHipRt
            {
                return buf.m_dev;
            }
        };

        //! The AsyncBufUniformCudaHipRt dimension getter trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DimType<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The AsyncBufUniformCudaHipRt memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct ElemType<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };
    } // namespace traits
    namespace extent
    {
        namespace traits
        {
            //! The AsyncBufUniformCudaHipRt extent get trait specialization.
            template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                ALPAKA_FN_HOST static auto getExtent(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx> const& extent) -> TIdx
                {
                    return extent.m_extentElements[TIdxIntegralConst::value];
                }
            };
        } // namespace traits
    } // namespace extent
    namespace traits
    {
        //! The AsyncBufUniformCudaHipRt native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx> const& buf)
                -> TElem const*
            {
                return buf.m_spMem.get();
            }
            ALPAKA_FN_HOST static auto getPtrNative(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>& buf) -> TElem*
            {
                return buf.m_spMem.get();
            }
        };
        //! The AsyncBufUniformCudaHipRt pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>, DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getPtrDev(
                AsyncBufUniformCudaHipRt<TElem, TDim, TIdx> const& buf,
                DevUniformCudaHipRt const& dev) -> TElem const*
            {
                if(dev == getDev(buf))
                {
                    return buf.m_spMem.get();
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
            ALPAKA_FN_HOST static auto getPtrDev(
                AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>& buf,
                DevUniformCudaHipRt const& dev) -> TElem*
            {
                if(dev == getDev(buf))
                {
                    return buf.m_spMem.get();
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
        };

        //! The CUDA/HIP 1D memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct AsyncBufAlloc<TElem, DimInt<1u>, TIdx, DevUniformCudaHipRt>
        {
            template<typename TQueue, typename TExtent>
            ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent)
                -> AsyncBufUniformCudaHipRt<TElem, DimInt<1u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const width = extent::getWidth(extent);
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(getDev(queue).m_iDevice));
                // Allocate the buffer on this device.
                void* memPtr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(MallocAsync)(
                    &memPtr,
                    static_cast<std::size_t>(widthBytes),
                    queue.m_spQueueImpl->m_UniformCudaHipQueue));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << " ptr: " << memPtr << std::endl;
#    endif
                return AsyncBufUniformCudaHipRt<TElem, DimInt<1u>, TIdx>(
                    queue,
                    reinterpret_cast<TElem*>(memPtr),
                    static_cast<TIdx>(widthBytes),
                    extent);
            }
        };
        //! The AsyncBufUniformCudaHipRt memory pinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Pin<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto pin(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always pinned, it can not be swapped out.
            }
        };
        //! The AsyncBufUniformCudaHipRt memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unpin<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto unpin(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always pinned, it can not be swapped out.
            }
        };
        //! The AsyncBufUniformCudaHipRt memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IsPinned<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto isPinned(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx> const&) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always pinned, it can not be swapped out.
                return true;
            }
        };
        //! The AsyncBufUniformCudaHipRt memory prepareForAsyncCopy trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct PrepareForAsyncCopy<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto prepareForAsyncCopy(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always ready for async copy
            }
        };

        //! The AsyncBufUniformCudaHipRt offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getOffset(AsyncBufUniformCudaHipRt<TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //! The AsyncBufUniformCudaHipRt idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IdxType<AsyncBufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };

    } // namespace traits
} // namespace alpaka

#    include <alpaka/mem/buf/uniformCudaHip/Copy.hpp>
#    include <alpaka/mem/buf/uniformCudaHip/Set.hpp>

#endif
