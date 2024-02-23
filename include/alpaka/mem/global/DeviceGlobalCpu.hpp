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
        //using TypeC = std::remove_all_extents_t<TViewDstFwd>;
        using Type = std::remove_const_t<std::remove_all_extents_t<TViewDstFwd>>;
        auto extent = getExtents(viewSrc);
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<decltype(extent)>, alpaka::Idx<decltype(extent)>>(
            //const_cast<std::remove_const_t<Type*>>(reinterpret_cast<Type*>(&viewDst)),
            reinterpret_cast<Type*>(const_cast<std::remove_const_t<TViewDstFwd>*>(&viewDst)),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    template<typename TViewSrc, typename TViewDstFwd, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(TQueue& queue, TViewDstFwd&& viewDst, alpaka::DevGlobal<TViewSrc>& viewSrc) -> void
    {
        using Type = std::remove_all_extents_t<TViewSrc>;
        auto extent = getExtents(viewDst);
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<decltype(extent)>, alpaka::Idx<decltype(extent)>>(
            reinterpret_cast<Type*>(&viewSrc),
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
        using Type = std::remove_const_t<std::remove_all_extents_t<TViewDstFwd>>;
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(const_cast<std::remove_const_t<TViewDstFwd>*>(&viewDst)),
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
        using Type = std::remove_all_extents_t<TViewSrc>;
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(&viewSrc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), view, extent));
    }
} // namespace alpaka
