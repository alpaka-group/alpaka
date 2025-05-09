/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/sycl/MutBufGenericSycl.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <memory>
#include <type_traits>

// TODO: delete
// #define ALPAKA_ACC_SYCL_ENABLED 1

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        //! The Sycl memory buffer implementation.
        template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
        class BufSyclImpl final
        {
            static_assert(
                !std::is_const_v<TElem>,
                "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
                "elements!");
            static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        public:
            template<typename TExtent>
            ALPAKA_FN_HOST BufSyclImpl(
                DevGenericSycl<TTag> dev,
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

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem)
                          << std::endl;
#    endif
            }

            BufSyclImpl(BufSyclImpl&&) = delete;
            auto operator=(BufSyclImpl&&) -> BufSyclImpl& = delete;

            ALPAKA_FN_HOST ~BufSyclImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // NOTE: m_pMem is allowed to be a nullptr here.
                m_deleter(m_pMem);
            }

        private:
            DevGenericSycl<TTag> m_dev;
            Vec<TDim, TIdx> m_extentElements;
            TElem* const m_pMem;
            std::function<void(TElem*)> m_deleter;
        };
    } // namespace detail

    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class ConstBufGenericSycl : public internal::ViewAccessOps<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
    public:
        //! Constructor
        template<typename TExtent, typename Deleter>
        ConstBufGenericSycl(DevGenericSycl<TTag> const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : m_dev{dev}
            , m_extentElements{getExtentVecEnd<TDim>(extent)}
            , m_spMem(pMem, std::move(deleter))
            , m_spBuf{std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");

            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

        //! Constructor for a ConstBuf from a non-ConstBuf
        ALPAKA_FN_HOST ConstBufGenericSycl(
            MutBufGenericSycl<ConstBufGenericSycl, TElem, DevGenericSycl<TTag>, TElem, TDim, TIdx> const& buf)
            : m_dev{trait::GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>::getDev(buf)}
            , m_extentElements{}
            , m_spMem{buf.m_spBufImpl}
        {
        }

    private:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufCpuImpl;

        friend alpaka::trait::GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetExtents<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TTag>
    using BufGenericSycl = MutBufGenericSycl<ConstBufGenericSycl, TElem, DevGenericSycl<TTag>, TElem, TDim, TIdx>;

} // namespace alpaka

namespace alpaka::trait
{
    //! The ConstBufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DevType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = DevGenericSycl<TTag>;
    };

    //! The ConstBufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        static auto getDev(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
        {
            return buf.m_dev;
        }
    };

    //! The ConstBufGenericSycl dimension getter trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DimType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TDim;
    };

    //! The ConstBufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct ElemType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TElem const;
    };

    //! The ConstBufGenericSycl extent get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetExtents<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        auto operator()(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) const
        {
            return buf.m_extentElements;
        }
    };

    //! The ConstBufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        static auto getPtrNative(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) -> TElem const*
        {
            return buf.m_spMem.get();
        }
    };

    //! The ConstBufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>
    {
        static auto getPtrDev(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf, DevGenericSycl<TTag> const& dev)
            -> TElem const*
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

    //! The SYCL memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct BufAlloc<TElem, TDim, TIdx, DevGenericSycl<TTag>>
    {
        template<typename TExtent>
        static auto allocBuf(DevGenericSycl<TTag> const& dev, TExtent const& extent)
            -> BufGenericSycl<TElem, TDim, TIdx, TTag>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            if constexpr(TDim::value == 0)
                std::cout << __func__ << " ewb: " << sizeof(TElem) << '\n';
            else if constexpr(TDim::value == 1)
            {
                auto const width = getWidth(extent);

                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << '\n';
            }
            else if constexpr(TDim::value == 2)
            {
                auto const width = getWidth(extent);
                auto const height = getHeight(extent);

                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ewb: " << widthBytes
                          << " pitch: " << widthBytes << '\n';
            }
            else if constexpr(TDim::value == 3)
            {
                auto const width = getWidth(extent);
                auto const height = getHeight(extent);
                auto const depth = getDepth(extent);

                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ed: " << depth
                          << " ewb: " << widthBytes << " pitch: " << widthBytes << '\n';
            }
#    endif

            auto const& [nativeDev, nativeContext] = dev.getNativeHandle();
            TElem* memPtr = sycl::malloc_device<TElem>(
                static_cast<std::size_t>(getExtentProduct(extent)),
                nativeDev,
                nativeContext);
            auto deleter = [ctx = nativeContext](TElem* ptr) { sycl::free(ptr, ctx); };

            return BufGenericSycl<TElem, TDim, TIdx, TTag>(dev, memPtr, std::move(deleter), extent);
        }
    };

    //! The BufGenericSycl stream-ordered memory allocation capability trait specialization.
    template<typename TDim, concepts::Tag TTag>
    struct HasAsyncBufSupport<TDim, DevGenericSycl<TTag>> : std::false_type
    {
    };

    //! The ConstBufGenericSycl offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetOffsets<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        auto operator()(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const&) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The pinned/mapped memory allocation trait specialization for the SYCL devices.
    template<concepts::Tag TTag, typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped<PlatformGenericSycl<TTag>, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        static auto allocMappedBuf(
            DevCpu const& host,
            PlatformGenericSycl<TTag> const& platform,
            TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Allocate SYCL page-locked memory on the host, mapped into the SYCL platform's address space and
            // accessible to all devices in the SYCL platform.
            auto ctx = platform.syclContext();
            TElem* memPtr = sycl::malloc_host<TElem>(static_cast<std::size_t>(getExtentProduct(extent)), ctx);
            auto deleter = [ctx](TElem* ptr) { sycl::free(ptr, ctx); };

            return BufCpu<TElem, TDim, TIdx>(host, memPtr, std::move(deleter), extent);
        }
    };

    //! The pinned/mapped memory allocation capability trait specialization.
    template<concepts::Tag TTag>
    struct HasMappedBufSupport<PlatformGenericSycl<TTag>> : public std::true_type
    {
    };

    //! The ConstBufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct IdxType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on SYCL device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevGenericSycl<TTag>>
    {
        static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevGenericSycl<TTag> const&) -> TElem const*
        {
            return getPtrNative(buf);
        }
    };
} // namespace alpaka::trait

#    include "alpaka/mem/buf/sycl/Copy.hpp"
#    include "alpaka/mem/buf/sycl/Set.hpp"

#endif
