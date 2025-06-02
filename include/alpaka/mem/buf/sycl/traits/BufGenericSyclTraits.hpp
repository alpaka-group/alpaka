/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/sycl/BufGenericSycl.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka::trait
{
    //! The SYCL device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct BufType<DevGenericSycl<TTag>, TElem, TDim, TIdx>
    {
        using type = BufGenericSycl<TElem, TDim, TIdx, TTag>;
    };

    //! The BufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DevType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = DevGenericSycl<TTag>;
    };

    //! The BufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetDev<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getDev(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) -> DevGenericSycl<TTag>
        {
            return buf.m_spBufImpl->m_dev;
        }
    };

    //! The BufGenericSycl dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DimType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TDim;
    };

    //! The BufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct ElemType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TElem;
    };

    //! The BufGenericSycl width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetExtents<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
        {
            return buf.m_spBufImpl->m_extentElements;
        }
    };

    //! The BufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrNative<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) -> TElem const*
        {
            return buf.m_spBufImpl->m_pMem;
        }

        ALPAKA_FN_HOST static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TTag>& buf) -> TElem*
        {
            return buf.m_spBufImpl->m_pMem;
        }
    };

    //! The BufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrDev<BufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf,
            DevGenericSycl<TTag> const& dev) -> TElem const*
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

        ALPAKA_FN_HOST static auto getPtrDev(
            BufGenericSycl<TElem, TDim, TIdx, TTag>& buf,
            DevGenericSycl<TTag> const& dev) -> TElem*
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

    //! The BufGenericSycl offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetOffsets<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(BufGenericSycl<TElem, TDim, TIdx, TTag> const& /*buf*/) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The BufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct IdxType<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for Sycl buffers.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct MakeConstBuf<BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
            -> ConstBufGenericSycl<TElem, TDim, TIdx, TTag>
        {
            return ConstBufGenericSycl<TElem, TDim, TIdx, TTag>(buf);
        }

        ALPAKA_FN_HOST static auto makeConstBuf(BufGenericSycl<TElem, TDim, TIdx, TTag>&& buf)
            -> ConstBufGenericSycl<TElem, TDim, TIdx, TTag>
        {
            return ConstBufGenericSycl<TElem, TDim, TIdx, TTag>(std::move(buf));
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

    //! The pinned/mapped memory allocation capability trait specialization.
    template<concepts::Tag TTag>
    struct HasMappedBufSupport<PlatformGenericSycl<TTag>> : public std::true_type
    {
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

} // namespace alpaka::trait

#endif
