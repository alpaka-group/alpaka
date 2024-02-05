/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/queue/sycl/QueueGenericSyclBase.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    using sycl::ext::oneapi::experimental::device_global;

    // from device to host
    template<typename TDev, bool TBlocking, typename TViewDst, typename TViewSrc>
    ALPAKA_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        TViewDst&& viewDst,
        device_global<TViewSrc> const& viewSrc)
    {
        queue.getNativeHandle().memcpy(reinterpret_cast<void*>(getPtrNative(viewDst)), viewSrc);
    }

    // from host to device
    template<typename TDev, bool TBlocking, typename TViewDst, typename TViewSrc>
    ALPAKA_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        device_global<TViewDst>& viewDst,
        TViewSrc const& viewSrc)
    {
        queue.getNativeHandle().memcpy(viewDst, reinterpret_cast<void const*>(getPtrNative(viewSrc)));
    }

    // from device to host
    template<typename TDev, bool TBlocking, typename TViewDst, typename TViewSrc, typename TExtent>
    ALPAKA_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        TViewDst&& viewDst,
        device_global<TViewSrc> const& viewSrc,
        TExtent extent)
    {
        using Elem = alpaka::Elem<std::remove_reference_t<TViewDst>>;
        auto size = static_cast<std::size_t>(getHeight(extent)) * static_cast<std::size_t>(getDepth(extent))
                    * static_cast<std::size_t>(getWidth(extent)) * sizeof(Elem);
        queue.getNativeHandle().memcpy(reinterpret_cast<void*>(getPtrNative(viewDst)), viewSrc, size);
    }

    // from host to device
    template<typename TDev, bool TBlocking, typename TViewDst, typename TViewSrc, typename TExtent>
    ALPAKA_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        device_global<TViewDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent extent)
    {
        using Elem = alpaka::Elem<TViewSrc>;
        auto size = static_cast<std::size_t>(getHeight(extent)) * static_cast<std::size_t>(getDepth(extent))
                    * static_cast<std::size_t>(getWidth(extent)) * sizeof(Elem);
        queue.getNativeHandle().memcpy(viewDst, reinterpret_cast<void const*>(getPtrNative(viewSrc)), size);
    }
} // namespace alpaka
#endif
