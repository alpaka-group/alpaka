/* Copyright 2023 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/mem/buf/BufCpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/view/Accessor.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <memory>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    class BufGenericSycl : public internal::ViewAccessOps<BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
    public:
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        //! Constructor
        template<typename TExtent, typename Deleter>
        BufGenericSycl(DevGenericSycl<TPltf> const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : m_dev{dev}
            , m_extentElements{getExtentVecEnd<TDim>(extent)}
            , m_spMem(pMem, std::move(deleter))
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

        DevGenericSycl<TPltf> m_dev;
        Vec<TDim, TIdx> m_extentElements;
        std::shared_ptr<TElem> m_spMem;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The BufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct DevType<BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        using type = DevGenericSycl<TPltf>;
    };

    //! The BufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetDev<BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        static auto getDev(BufGenericSycl<TElem, TDim, TIdx, TPltf> const& buf)
        {
            return buf.m_dev;
        }
    };

    //! The BufGenericSycl dimension getter trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct DimType<BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        using type = TDim;
    };

    //! The BufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct ElemType<BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        using type = TElem;
    };

    //! The BufGenericSycl extent get trait specialization.
    template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetExtent<TIdxIntegralConst, BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        static_assert(TDim::value > TIdxIntegralConst::value, "Requested dimension out of bounds");

        static auto getExtent(BufGenericSycl<TElem, TDim, TIdx, TPltf> const& buf) -> TIdx
        {
            return buf.m_extentElements[TIdxIntegralConst::value];
        }
    };

    //! The BufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetPtrNative<BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TPltf> const& buf) -> TElem const*
        {
            return buf.m_spMem.get();
        }

        static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TPltf>& buf) -> TElem*
        {
            return buf.m_spMem.get();
        }
    };

    //! The BufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetPtrDev<BufGenericSycl<TElem, TDim, TIdx, TPltf>, DevGenericSycl<TPltf>>
    {
        static auto getPtrDev(BufGenericSycl<TElem, TDim, TIdx, TPltf> const& buf, DevGenericSycl<TPltf> const& dev)
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

        static auto getPtrDev(BufGenericSycl<TElem, TDim, TIdx, TPltf>& buf, DevGenericSycl<TPltf> const& dev)
            -> TElem*
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
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct BufAlloc<TElem, TDim, TIdx, DevGenericSycl<TPltf>>
    {
        template<typename TExtent>
        static auto allocBuf(DevGenericSycl<TPltf> const& dev, TExtent const& extent)
            -> BufGenericSycl<TElem, TDim, TIdx, TPltf>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            if constexpr(TDim::value == 0 || TDim::value == 1)
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
            // captured structured bindings are a C++20 extension
            // auto deleter = [nativeContext](TElem* ptr) { sycl::free(ptr, nativeContext); };
            auto deleter = [&dev](TElem* ptr) { sycl::free(ptr, dev.getNativeHandle().second); };

            return BufGenericSycl<TElem, TDim, TIdx, TPltf>(dev, memPtr, std::move(deleter), extent);
        }
    };

    //! The BufGenericSycl stream-ordered memory allocation capability trait specialization.
    template<typename TDim, typename TPltf>
    struct HasAsyncBufSupport<TDim, DevGenericSycl<TPltf>> : std::false_type
    {
    };

    //! The BufGenericSycl offset get trait specialization.
    template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetOffset<TIdxIntegralConst, BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        static auto getOffset(BufGenericSycl<TElem, TDim, TIdx, TPltf> const&) -> TIdx
        {
            return 0u;
        }
    };

    //! The pinned/mapped memory allocation trait specialization for the SYCL devices.
    template<typename TPltf, typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped
    {
        template<typename TExtent>
        static auto allocMappedBuf(DevCpu const& host, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Allocate SYCL page-locked memory on the host, mapped into the TPltf address space and
            // accessible to all devices in the TPltf.
            auto ctx = TPltf::syclContext();
            TElem* memPtr = sycl::malloc_host<TElem>(static_cast<std::size_t>(getExtentProduct(extent)), ctx);
            auto deleter = [ctx](TElem* ptr) { sycl::free(ptr, ctx); };

            return BufCpu<TElem, TDim, TIdx>(host, memPtr, std::move(deleter), extent);
        }
    };

    //! The BufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct IdxType<BufGenericSycl<TElem, TDim, TIdx, TPltf>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on SYCL device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevGenericSycl<TPltf>>
    {
        static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevGenericSycl<TPltf> const&) -> TElem const*
        {
            return getPtrNative(buf);
        }
        static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevGenericSycl<TPltf> const&) -> TElem*
        {
            return getPtrNative(buf);
        }
    };
} // namespace alpaka::trait

#    include "alpaka/mem/buf/sycl/Accessor.hpp"
#    include "alpaka/mem/buf/sycl/Copy.hpp"
#    include "alpaka/mem/buf/sycl/Set.hpp"

#endif
