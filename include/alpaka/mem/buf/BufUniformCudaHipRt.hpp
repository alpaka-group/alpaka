/* Copyright 2022 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 * Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/mem/view/ViewAccessOps.hpp>
#    include <alpaka/meta/DependentFalseType.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <functional>
#    include <memory>
#    include <type_traits>

namespace alpaka
{
    class DevUniformCudaHipRt;

    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    //! The CUDA/HIP memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRt : public internal::ViewAccessOps<BufUniformCudaHipRt<TElem, TDim, TIdx>>
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
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufUniformCudaHipRt(
            DevUniformCudaHipRt const& dev,
            TElem* const pMem,
            Deleter deleter,
            TIdx const& pitchBytes,
            TExtent const& extent)
            : m_dev(dev)
            , m_extentElements(extent::getExtentVecEnd<TDim>(extent))
            , m_spMem(pMem, std::move(deleter))
            , m_pitchBytes(pitchBytes)
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

    public:
        DevUniformCudaHipRt m_dev;
        Vec<TDim, TIdx> m_extentElements;
        std::shared_ptr<TElem> m_spMem;
        TIdx m_pitchBytes;
    };

    namespace traits
    {
        //! The BufUniformCudaHipRt device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = DevUniformCudaHipRt;
        };
        //! The BufUniformCudaHipRt device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetDev<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(BufUniformCudaHipRt<TElem, TDim, TIdx> const& buf) -> DevUniformCudaHipRt
            {
                return buf.m_dev;
            }
        };

        //! The BufUniformCudaHipRt dimension getter trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DimType<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The BufUniformCudaHipRt memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct ElemType<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };
    } // namespace traits
    namespace extent
    {
        namespace traits
        {
            //! The BufUniformCudaHipRt extent get trait specialization.
            template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                BufUniformCudaHipRt<TElem, TDim, TIdx>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                ALPAKA_FN_HOST static auto getExtent(BufUniformCudaHipRt<TElem, TDim, TIdx> const& extent) -> TIdx
                {
                    return extent.m_extentElements[TIdxIntegralConst::value];
                }
            };
        } // namespace traits
    } // namespace extent
    namespace traits
    {
        //! The BufUniformCudaHipRt native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return buf.m_spMem.get();
            }
            ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TElem, TDim, TIdx>& buf) -> TElem*
            {
                return buf.m_spMem.get();
            }
        };
        //! The BufUniformCudaHipRt pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufUniformCudaHipRt<TElem, TDim, TIdx>, DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getPtrDev(
                BufUniformCudaHipRt<TElem, TDim, TIdx> const& buf,
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
                BufUniformCudaHipRt<TElem, TDim, TIdx>& buf,
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
        //! The BufUniformCudaHipRt pitch get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPitchBytes<DimInt<TDim::value - 1u>, BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPitchBytes(BufUniformCudaHipRt<TElem, TDim, TIdx> const& buf) -> TIdx
            {
                return buf.m_pitchBytes;
            }
        };

        //! The CUDA/HIP scalar memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct BufAlloc<TElem, DimInt<0u>, TIdx, DevUniformCudaHipRt>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevUniformCudaHipRt const& dev, [[maybe_unused]] TExtent const& extent)
                -> BufUniformCudaHipRt<TElem, DimInt<0u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                using Scalar = alpaka::Vec<alpaka::DimInt<0u>, TIdx>;
                ALPAKA_ASSERT(Scalar{} == extent);

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.m_iDevice));
                // Allocate the buffer on this device.
                void* memPtr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(Malloc)(&memPtr, sizeof(TElem)));
                // Prepare a deleter for this device
                auto deleter = [](TElem* ptr)
                {
                    // Free the buffer.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(Free)(reinterpret_cast<void*>(ptr)));
                };
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << 1u << " ewb: " << sizeof(TElem) << " ptr: " << memPtr << std::endl;
#    endif
                return BufUniformCudaHipRt<TElem, DimInt<0u>, TIdx>(
                    dev,
                    reinterpret_cast<TElem*>(memPtr),
                    std::move(deleter),
                    static_cast<TIdx>(sizeof(TElem)),
                    Scalar{});
            }
        };
        //! The CUDA/HIP 1D memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct BufAlloc<TElem, DimInt<1u>, TIdx, DevUniformCudaHipRt>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevUniformCudaHipRt const& dev, TExtent const& extent)
                -> BufUniformCudaHipRt<TElem, DimInt<1u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const width = extent::getWidth(extent);
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.m_iDevice));
                // Allocate the buffer on this device.
                void* memPtr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(Malloc)(&memPtr, static_cast<std::size_t>(widthBytes)));
                // Prepare a deleter for this device
                auto deleter = [](TElem* ptr)
                {
                    // Free the buffer.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(Free)(reinterpret_cast<void*>(ptr)));
                };
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << " ptr: " << memPtr << std::endl;
#    endif
                return BufUniformCudaHipRt<TElem, DimInt<1u>, TIdx>(
                    dev,
                    reinterpret_cast<TElem*>(memPtr),
                    std::move(deleter),
                    static_cast<TIdx>(widthBytes),
                    extent);
            }
        };
        //! The CUDA/HIP 2D memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct BufAlloc<TElem, DimInt<2u>, TIdx, DevUniformCudaHipRt>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevUniformCudaHipRt const& dev, TExtent const& extent)
                -> BufUniformCudaHipRt<TElem, DimInt<2u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const width = extent::getWidth(extent);
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                auto const height = extent::getHeight(extent);

                void* memPtr = nullptr;
                std::size_t pitchBytes = 0u;
#    ifdef ALPAKA_ACC_GPU_HIP_ENABLED
                // FIXME: HIP cannot handle zero-size input (throws Unknown Error)
                if(width != 0 && height != 0)
#    endif
                {
                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.m_iDevice));
                    // Allocate the buffer on this device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(MallocPitch)(
                        &memPtr,
                        &pitchBytes,
                        static_cast<std::size_t>(widthBytes),
                        static_cast<std::size_t>(height)));
                    ALPAKA_ASSERT(pitchBytes >= static_cast<std::size_t>(widthBytes) || (width * height) == 0);
                }
                // Prepare a deleter for this device
                auto deleter = [](TElem* ptr)
                {
                    // Free the buffer.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(Free)(reinterpret_cast<void*>(ptr)));
                };
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ewb: " << widthBytes
                          << " ptr: " << memPtr << " pitch: " << pitchBytes << std::endl;
#    endif
                return BufUniformCudaHipRt<TElem, DimInt<2u>, TIdx>(
                    dev,
                    reinterpret_cast<TElem*>(memPtr),
                    std::move(deleter),
                    static_cast<TIdx>(pitchBytes),
                    extent);
            }
        };
        //! The CUDA/HIP 3D memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct BufAlloc<TElem, DimInt<3u>, TIdx, DevUniformCudaHipRt>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevUniformCudaHipRt const& dev, TExtent const& extent)
                -> BufUniformCudaHipRt<TElem, DimInt<3u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_API_PREFIX(Extent)
                const extentVal = ALPAKA_PP_CONCAT(make_, ALPAKA_API_PREFIX(Extent))(
                    static_cast<std::size_t>(extent::getWidth(extent) * static_cast<TIdx>(sizeof(TElem))),
                    static_cast<std::size_t>(extent::getHeight(extent)),
                    static_cast<std::size_t>(extent::getDepth(extent)));

                ALPAKA_API_PREFIX(PitchedPtr) pitchedPtrVal;
                pitchedPtrVal.ptr = nullptr;
#    ifdef ALPAKA_ACC_GPU_HIP_ENABLED
                pitchedPtrVal.pitch = 0u;
                // FIXME: HIP cannot handle zero-size input
                if(extentVal.width != 0 && extentVal.height != 0 && extentVal.depth != 0)
#    endif
                {
                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.m_iDevice));
                    // Allocate the buffer on this device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(Malloc3D)(&pitchedPtrVal, extentVal));
                }
                // Prepare a deleter for this device
                auto deleter = [](TElem* ptr)
                {
                    // Free the buffer.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(Free)(reinterpret_cast<void*>(ptr)));
                };
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << extent::getWidth(extent) << " eh: " << extentVal.height
                          << " ed: " << extentVal.depth << " ewb: " << extentVal.width << " ptr: " << pitchedPtrVal.ptr
                          << " pitch: " << pitchedPtrVal.pitch << " wb: " << pitchedPtrVal.xsize
                          << " h: " << pitchedPtrVal.ysize << std::endl;
#    endif
                return BufUniformCudaHipRt<TElem, DimInt<3u>, TIdx>(
                    dev,
                    reinterpret_cast<TElem*>(pitchedPtrVal.ptr),
                    std::move(deleter),
                    static_cast<TIdx>(pitchedPtrVal.pitch),
                    extent);
            }
        };

        //! The CUDA/HIP stream-ordered scalar memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct AsyncBufAlloc<TElem, DimInt<0u>, TIdx, DevUniformCudaHipRt>
        {
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && (BOOST_LANG_CUDA < BOOST_VERSION_NUMBER(11, 2, 0))
            static_assert(
                meta::DependentFalseType<TElem>::value,
                "Support for stream-ordered memory buffers requires CUDA 11.2 or higher.");
#    endif
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
            static_assert(
                meta::DependentFalseType<TElem>::value,
                "HIP devices do not support stream-ordered memory buffers.");
#    endif

            template<typename TQueue, typename TExtent>
            ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, [[maybe_unused]] TExtent const& extent)
                -> BufUniformCudaHipRt<TElem, DimInt<0u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                using Scalar = alpaka::Vec<alpaka::DimInt<0u>, TIdx>;
                ALPAKA_ASSERT(Scalar{} == extent);

                // Set the current device.
                DevUniformCudaHipRt const& dev = getDev(queue);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.m_iDevice));
                // Allocate the buffer on this device.
                void* memPtr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(
                    MallocAsync)(&memPtr, sizeof(TElem), queue.m_spQueueImpl->m_UniformCudaHipQueue));
                // Prepare a deleter for this device
                auto deleter = [queue = std::move(queue)](TElem* ptr)
                {
                    // Free the buffer.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(
                        FreeAsync)(reinterpret_cast<void*>(ptr), queue.m_spQueueImpl->m_UniformCudaHipQueue));
                };
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << 1u << " ewb: " << sizeof(TElem) << " ptr: " << memPtr << std::endl;
#    endif
                return BufUniformCudaHipRt<TElem, DimInt<0u>, TIdx>(
                    dev,
                    reinterpret_cast<TElem*>(memPtr),
                    std::move(deleter),
                    static_cast<TIdx>(sizeof(TElem)),
                    Scalar{});
            }
        };
        //! The CUDA/HIP stream-ordered 1D memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct AsyncBufAlloc<TElem, DimInt<1u>, TIdx, DevUniformCudaHipRt>
        {
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && (CUDA_VERSION < 11020)
            static_assert(
                meta::DependentFalseType<TElem>::value,
                "Support for stream-ordered memory buffers requires CUDA 11.2 or higher.");
#    endif
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
            static_assert(
                meta::DependentFalseType<TElem>::value,
                "HIP devices do not support stream-ordered memory buffers.");
#    endif

            template<typename TQueue, typename TExtent>
            ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent)
                -> BufUniformCudaHipRt<TElem, DimInt<1u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const width = extent::getWidth(extent);
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));

                // Set the current device.
                DevUniformCudaHipRt const& dev = getDev(queue);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.m_iDevice));
                // Allocate the buffer on this device.
                void* memPtr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(MallocAsync)(
                    &memPtr,
                    static_cast<std::size_t>(widthBytes),
                    queue.m_spQueueImpl->m_UniformCudaHipQueue));
                // Prepare a deleter for this device
                auto deleter = [queue = std::move(queue)](TElem* ptr)
                {
                    // Free the buffer.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(
                        FreeAsync)(reinterpret_cast<void*>(ptr), queue.m_spQueueImpl->m_UniformCudaHipQueue));
                };
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << " ptr: " << memPtr << std::endl;
#    endif
                return BufUniformCudaHipRt<TElem, DimInt<1u>, TIdx>(
                    dev,
                    reinterpret_cast<TElem*>(memPtr),
                    std::move(deleter),
                    static_cast<TIdx>(widthBytes),
                    extent);
            }
        };
        //! The CUDA/HIP stream-ordered ND memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct AsyncBufAlloc<TElem, TDim, TIdx, DevUniformCudaHipRt>
        {
            static_assert(
                meta::DependentFalseType<TElem>::value,
                "HIP/CUDA devices support only one-dimensional stream-ordered memory buffers.");
        };
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && (CUDA_VERSION >= 11020)
        //! The CUDA/HIP stream-ordered scalar memory allocation capability trait specialization.
        template<>
        struct HasAsyncBufSupport<DimInt<0u>, DevUniformCudaHipRt> : public std::true_type
        {
        };
        //! The CUDA/HIP stream-ordered 1D memory allocation capability trait specialization.
        template<>
        struct HasAsyncBufSupport<DimInt<1u>, DevUniformCudaHipRt> : public std::true_type
        {
        };
#    endif

        //! The BufUniformCudaHipRt CUDA/HIP device memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Map<BufUniformCudaHipRt<TElem, TDim, TIdx>, DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto map(
                BufUniformCudaHipRt<TElem, TDim, TIdx> const& buf,
                DevUniformCudaHipRt const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                {
                    throw std::runtime_error(
                        "Mapping memory from one CUDA/HIP device into an other CUDA/HIP device not implemented!");
                }
                // If it is already the same device, nothing has to be mapped.
            }
        };
        //! The BufUniformCudaHipRt CUDA/HIP device memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unmap<BufUniformCudaHipRt<TElem, TDim, TIdx>, DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto unmap(
                BufUniformCudaHipRt<TElem, TDim, TIdx> const& buf,
                DevUniformCudaHipRt const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                {
                    throw std::runtime_error("Unmapping memory mapped from one CUDA/HIP device into an other CUDA/HIP "
                                             "device not implemented!");
                }
                // If it is already the same device, nothing has to be unmapped.
            }
        };
        //! The BufUniformCudaHipRt memory pinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Pin<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto pin(BufUniformCudaHipRt<TElem, TDim, TIdx>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always pinned, it can not be swapped out.
            }
        };
        //! The BufUniformCudaHipRt memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unpin<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto unpin(BufUniformCudaHipRt<TElem, TDim, TIdx>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always pinned, it can not be swapped out.
            }
        };
        //! The BufUniformCudaHipRt memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IsPinned<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto isPinned(BufUniformCudaHipRt<TElem, TDim, TIdx> const&) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always pinned, it can not be swapped out.
                return true;
            }
        };
        //! The BufUniformCudaHipRt memory prepareForAsyncCopy trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct PrepareForAsyncCopy<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto prepareForAsyncCopy(BufUniformCudaHipRt<TElem, TDim, TIdx>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // CUDA/HIP device memory is always ready for async copy
            }
        };

        //! The BufUniformCudaHipRt offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getOffset(BufUniformCudaHipRt<TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //! The BufUniformCudaHipRt idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IdxType<BufUniformCudaHipRt<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The BufCpu CUDA/HIP device memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Map<BufCpu<TElem, TDim, TIdx>, DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto map(BufCpu<TElem, TDim, TIdx>& buf, DevUniformCudaHipRt const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // If it is already the same device, nothing has to be mapped.
                if(getDev(buf) != dev)
                {
                    // cuda/hip-HostRegisterMapped:
                    //   Maps the allocation into the CUDA/HIP address space.The device pointer to the memory may be
                    //   obtained by calling cudaHostGetDevicePointer(). This feature is available only on GPUs with
                    //   compute capability greater than or equal to 1.1.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(HostRegister)(
                        const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                        extent::getExtentProduct(buf) * sizeof(Elem<BufCpu<TElem, TDim, TIdx>>),
                        ALPAKA_API_PREFIX(HostRegisterMapped)));
                }
            }
        };
        //! The BufCpu CUDA/HIP device memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unmap<BufCpu<TElem, TDim, TIdx>, DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto unmap(BufCpu<TElem, TDim, TIdx>& buf, DevUniformCudaHipRt const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                {
                    // Unmaps the memory range whose base address is specified by ptr, and makes it pageable again.
                    // \FIXME: If the memory has separately been pinned before we destroy the pinning state.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(HostUnregister)(
                        const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf)))));
                }
                // If it is already the same device, nothing has to be unmapped.
            }
        };

        //! The BufCpu pointer on CUDA/HIP device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevUniformCudaHipRt const&)
                -> TElem const*
            {
                // TODO: Check if the memory is mapped at all!
                TElem* pDev(nullptr);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(HostGetDevicePointer)(
                    &pDev,
                    const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                    0));

                return pDev;
            }
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevUniformCudaHipRt const&) -> TElem*
            {
                // TODO: Check if the memory is mapped at all!
                TElem* pDev(nullptr);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(HostGetDevicePointer)(&pDev, getPtrNative(buf), 0));

                return pDev;
            }
        };
    } // namespace traits
} // namespace alpaka

#    include <alpaka/mem/buf/uniformCudaHip/Copy.hpp>
#    include <alpaka/mem/buf/uniformCudaHip/Set.hpp>

#endif
