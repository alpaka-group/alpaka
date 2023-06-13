/* Copyright 2022 Jan Stephan
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
#include "alpaka/vec/Vec.hpp"

#include <memory>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <CL/sycl.hpp>

namespace alpaka
{
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    class BufGenericSycl
    {
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

    public:
        //! Constructor
        template<typename TExtent>
        BufGenericSycl(TDev const& dev, sycl::buffer<TElem, TDim::value> buffer, TExtent const& extent)
            : m_dev{dev}
            , m_extentElements{getExtentVecEnd<TDim>(extent)}
            , m_buffer{buffer}
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

        TDev m_dev;
        Vec<TDim, TIdx> m_extentElements;
        sycl::buffer<TElem, TDim::value> m_buffer;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The BufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct DevType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TDev;
    };

    //! The BufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetDev<BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static auto getDev(BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf)
        {
            return buf.m_dev;
        }
    };

    //! The BufGenericSycl dimension getter trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct DimType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TDim;
    };

    //! The BufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct ElemType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TElem;
    };

    //! The BufGenericSycl extent get trait specialization.
    template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetExtent<TIdxIntegralConst, BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static_assert(TDim::value > TIdxIntegralConst::value, "Requested dimension out of bounds");

        static auto getExtent(BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf) -> TIdx
        {
            return buf.m_extentElements[TIdxIntegralConst::value];
        }
    };

    //! The BufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetPtrNative<BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static_assert(
            !sizeof(TElem),
            "Accessing device-side pointers on the host is not supported by the SYCL back-end");

        static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TDev> const&) -> TElem const*
        {
            return nullptr;
        }

        static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TDev>&) -> TElem*
        {
            return nullptr;
        }
    };

    //! The BufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetPtrDev<BufGenericSycl<TElem, TDim, TIdx, TDev>, TDev>
    {
        static_assert(
            !sizeof(TElem),
            "Accessing device-side pointers on the host is not supported by the SYCL back-end");

        static auto getPtrDev(BufGenericSycl<TElem, TDim, TIdx, TDev> const&, TDev const&) -> TElem const*
        {
            return nullptr;
        }

        static auto getPtrDev(BufGenericSycl<TElem, TDim, TIdx, TDev>&, TDev const&) -> TElem*
        {
            return nullptr;
        }
    };

    //! The SYCL memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct BufAlloc<TElem, TDim, TIdx, DevGenericSycl<TPltf>>
    {
        template<typename TExtent>
        static auto allocBuf(DevGenericSycl<TPltf> const& dev, TExtent const& ext)
            -> BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            if constexpr(TDim::value == 0 || TDim::value == 1)
            {
                auto const width = getWidth(ext);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << '\n';
#    endif

                auto const range = sycl::range<1>{width};
                return {dev, sycl::buffer<TElem, 1>{range}, ext};
            }
            else if constexpr(TDim::value == 2)
            {
                auto const width = getWidth(ext);
                auto const height = getHeight(ext);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ewb: " << widthBytes
                          << " pitch: " << widthBytes << '\n';
#    endif

                auto const range = sycl::range<2>{width, height};
                return {dev, sycl::buffer<TElem, 2>{range}, ext};
            }
            else if constexpr(TDim::value == 3)
            {
                auto const width = getWidth(ext);
                auto const height = getHeight(ext);
                auto const depth = getDepth(ext);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ed: " << depth
                          << " ewb: " << widthBytes << " pitch: " << widthBytes << '\n';
#    endif

                auto const range = sycl::range<3>{width, height, depth};
                return {dev, sycl::buffer<TElem, 3>{range}, ext};
            }
        }
    };

    //! The BufGenericSycl offset get trait specialization.
    template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetOffset<TIdxIntegralConst, BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static auto getOffset(BufGenericSycl<TElem, TDim, TIdx, TDev> const&) -> TIdx
        {
            return 0u;
        }
    };

    //! The BufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct IdxType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on SYCL device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevGenericSycl<TPltf>>
    {
        static_assert(!sizeof(TElem), "Accessing host pointers on the device is not supported by the SYCL back-end");

        static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const&, DevGenericSycl<TPltf> const&) -> TElem const*
        {
            return nullptr;
        }

        static auto getPtrDev(BufCpu<TElem, TDim, TIdx>&, DevGenericSycl<TPltf> const&) -> TElem*
        {
            return nullptr;
        }
    };
} // namespace alpaka::trait

#    include "alpaka/mem/buf/sycl/Accessor.hpp"
#    include "alpaka/mem/buf/sycl/Copy.hpp"
#    include "alpaka/mem/buf/sycl/Set.hpp"

#endif
