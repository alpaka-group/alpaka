/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/cpu/Copy.hpp"
#include "alpaka/mem/view/ViewPlainPtr.hpp"

#include <type_traits>

// memcpy specialization for device global variables
namespace alpaka
{
    template<typename TViewSrc, typename TViewDstFwd, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(TQueue& queue, alpaka::DevGlobal<TViewDstFwd>& viewDst, TViewSrc const& viewSrc) -> void
    {
        typedef std::remove_all_extents_t<TViewDstFwd> T;
        auto extent = getExtents(viewSrc);
        auto view = alpaka::ViewPlainPtr<DevCpu, T, alpaka::Dim<decltype(extent)>, alpaka::Idx<decltype(extent)>>(
            reinterpret_cast<T*>(&viewDst),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    template<typename TViewSrc, typename TViewDstFwd, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(TQueue& queue, TViewDstFwd&& viewDst, alpaka::DevGlobal<TViewSrc>& viewSrc) -> void
    {
        typedef std::remove_all_extents_t<TViewSrc> T;
        auto extent = getExtents(viewDst);
        auto view = alpaka::ViewPlainPtr<DevCpu, T, alpaka::Dim<decltype(extent)>, alpaka::Idx<decltype(extent)>>(
            reinterpret_cast<T*>(&viewSrc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), view, extent));
    }

    template<typename TExtent, typename TViewSrc, typename TViewDstFwd, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        alpaka::DevGlobal<TViewDstFwd>& viewDst,
        TViewSrc const& viewSrc,
        TExtent const& extent) -> void
    {
        typedef std::remove_all_extents_t<TViewDstFwd> T;
        auto view = alpaka::ViewPlainPtr<DevCpu, T, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<T*>(&viewDst),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    template<typename TExtent, typename TViewSrc, typename TViewDstFwd, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        TViewDstFwd&& viewDst,
        alpaka::DevGlobal<TViewSrc>& viewSrc,
        TExtent const& extent) -> void
    {
        typedef std::remove_all_extents_t<TViewSrc> T;
        auto view = alpaka::ViewPlainPtr<DevCpu, T, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<T*>(&viewSrc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), view, extent));
    }
} // namespace alpaka
