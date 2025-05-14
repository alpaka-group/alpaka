/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/sycl/ConstBufGenericSycl.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka::trait
{
    //! The SYCL device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct BufType<DevGenericSycl<TTag>, TElem, TDim, TIdx>
    {
        using type = BufGenericSycl<TElem, TDim, TIdx, TTag>;
    };

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
        ALPAKA_FN_HOST static auto getDev(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
        {
            return buf.m_spBufSyclImpl->m_dev;
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
        ALPAKA_FN_HOST auto operator()(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) const
        {
            return buf.m_spBufSyclImpl->m_extentElements;
        }
    };

    //! The ConstBufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
            -> TElem const*
        {
            return buf.m_spBufSyclImpl->m_pMem;
        }
    };

    //! The ConstBufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf,
            DevGenericSycl<TTag> const& dev) -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spBufSyclImpl->m_pMem;
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
        ALPAKA_FN_HOST static auto allocBuf(DevGenericSycl<TTag> const& dev, TExtent const& extent)
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
        ALPAKA_FN_HOST auto operator()(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const&) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The pinned/mapped memory allocation trait specialization for the SYCL devices.
    template<concepts::Tag TTag, typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped<PlatformGenericSycl<TTag>, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocMappedBuf(
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

#endif
