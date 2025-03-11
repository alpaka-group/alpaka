/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiCudaRt.hpp"
#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/core/Vectorize.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/alloc/AllocCpuAligned.hpp"
#include "alpaka/mem/buf/GenericMutableBuf.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/platform/PlatformCpu.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace detail
    {
        //! The CPU memory buffer.
        template<typename TElem, typename TDim, typename TIdx>
        class BufCpuImpl final
        {
            static_assert(
                !std::is_const_v<TElem>,
                "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
                "elements!");
            static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        public:
            template<typename TExtent>
            ALPAKA_FN_HOST BufCpuImpl(
                DevCpu dev,
                TElem* pMem,
                std::function<void(TElem*)> deleter,
                TExtent const& extent) noexcept
                : m_dev(std::move(dev))
                , m_extentElements(getExtentVecEnd<TDim>(extent))
                , m_pMem(pMem)
                , m_deleter(std::move(deleter))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    TDim::value == Dim<TExtent>::value,
                    "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                    "identical!");
                static_assert(
                    std::is_same_v<TIdx, Idx<TExtent>>,
                    "The idx type of TExtent and the TIdx template parameter have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem)
                          << std::endl;
#endif
            }

            BufCpuImpl(BufCpuImpl&&) = delete;
            auto operator=(BufCpuImpl&&) -> BufCpuImpl& = delete;

            ALPAKA_FN_HOST ~BufCpuImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // NOTE: m_pMem is allowed to be a nullptr here.
                m_deleter(m_pMem);
            }

        public:
            DevCpu const m_dev;
            Vec<TDim, TIdx> const m_extentElements;
            TElem* const m_pMem;
            std::function<void(TElem*)> m_deleter;
        };
    } // namespace detail

    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class GenericCpuBuf : public internal::ViewAccessOps<GenericCpuBuf<TElem, TDim, TIdx>>
    {
    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST GenericCpuBuf(DevCpu const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
            : m_spBufCpuImpl{
                std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
        {
        }

        ALPAKA_FN_HOST GenericCpuBuf(
            GenericBuf<GenericCpuBuf, detail::BufCpuImpl<TElem, TDim, TIdx>, DevCpu, TElem, TDim, TIdx> const& buf)
            : m_spBufCpuImpl{buf.m_spBufImpl}
        {
        }

    private:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufCpuImpl;

        friend alpaka::trait::GetDev<GenericCpuBuf<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetExtents<GenericCpuBuf<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrNative<GenericCpuBuf<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrDev<GenericCpuBuf<TElem, TDim, TIdx>, DevCpu>;
    };

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufCpu = GenericCpuBuf<TElem, TDim, TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufCpu = GenericBuf<GenericCpuBuf, alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>, DevCpu, TElem, TDim, TIdx>;

    namespace trait
    {
        //! The CPU device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<DevCpu, TElem, TDim, TIdx>
        {
            using type = BufCpu<TElem, TDim, TIdx>;
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
            // const qualify the element type of the inner view
            using type = TElem const;
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

        //! The ConstBufCpu native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<ConstBufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(ConstBufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return buf.m_spBufCpuImpl->m_pMem;
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
                    return buf.m_spBufCpuImpl->m_pMem;
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
        };

        //! The BufCpu memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufAlloc<TElem, TDim, TIdx, DevCpu>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevCpu const& dev, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // If ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT is defined, positive, and a power of 2, use it as the
                // default alignment for host memory allocations. Otherwise, the alignment is chosen to enable
                // optimal performance dependant on the target architecture.
#if defined(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT)
                static_assert(
                    ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT > 0
                        && ((ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT & (ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT - 1)) == 0),
                    "If defined, ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT must be a power of 2.");
                constexpr std::size_t alignment = static_cast<std::size_t>(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT);
#else
                constexpr std::size_t alignment = core::vectorization::defaultAlignment;
#endif
                // alpaka::AllocCpuAligned is stateless
                using Allocator = AllocCpuAligned<std::integral_constant<std::size_t, alignment>>;
                static_assert(std::is_empty_v<Allocator>, "AllocCpuAligned is expected to be stateless");
                auto* memPtr = alpaka::malloc<TElem>(Allocator{}, static_cast<std::size_t>(getExtentProduct(extent)));
                auto deleter = [](TElem* ptr) { alpaka::free(Allocator{}, ptr); };

                return BufCpu<TElem, TDim, TIdx>(dev, memPtr, std::move(deleter), extent);
            }
        };

        //! The BufCpu stream-ordered memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct AsyncBufAlloc<TElem, TDim, TIdx, DevCpu>
        {
            template<typename TQueue, typename TExtent>
            ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    std::is_same_v<Dev<TQueue>, DevCpu>,
                    "The BufCpu buffer can only be used with a queue on a DevCpu device!");
                DevCpu const& dev = getDev(queue);

                // If ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT is defined, positive, and a power of 2, use it as the
                // default alignment for host memory allocations. Otherwise, the alignment is chosen to enable
                // optimal performance dependant on the target architecture.
#if defined(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT)
                static_assert(
                    ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT > 0
                        && ((ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT & (ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT - 1)) == 0),
                    "If defined, ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT must be a power of 2.");
                constexpr std::size_t alignment = static_cast<std::size_t>(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT);
#else
                constexpr std::size_t alignment = core::vectorization::defaultAlignment;
#endif
                // alpaka::AllocCpuAligned is stateless
                using Allocator = AllocCpuAligned<std::integral_constant<std::size_t, alignment>>;
                static_assert(std::is_empty_v<Allocator>, "AllocCpuAligned is expected to be stateless");
                auto* memPtr = alpaka::malloc<TElem>(Allocator{}, static_cast<std::size_t>(getExtentProduct(extent)));
                auto deleter = [l_queue = std::move(queue)](TElem* ptr) mutable
                {
                    alpaka::enqueue(
                        l_queue,
                        [ptr]()
                        {
                            // free the memory
                            alpaka::free(Allocator{}, ptr);
                        });
                };

                return BufCpu<TElem, TDim, TIdx>(dev, memPtr, std::move(deleter), extent);
            }
        };

        //! The ConstBufCpu stream-ordered memory allocation capability trait specialization.
        template<typename TDim>
        struct HasAsyncBufSupport<TDim, DevCpu> : public std::true_type
        {
        };

        //! The pinned/mapped memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufAllocMapped<PlatformCpu, TElem, TDim, TIdx>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocMappedBuf(
                DevCpu const& host,
                PlatformCpu const& /*platform*/,
                TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
            {
                // Allocate standard host memory.
                return allocBuf<TElem, TIdx>(host, extent);
            }
        };

        //! The pinned/mapped memory allocation capability trait specialization.
        template<>
        struct HasMappedBufSupport<PlatformCpu> : public std::true_type
        {
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

#include "alpaka/mem/buf/cpu/Copy.hpp"
#include "alpaka/mem/buf/cpu/Set.hpp"
