/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/queue/cuda_hip/QueueUniformCudaHipRt.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    // from device to host
    template<typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        TViewDst& viewDst,
        alpaka::DevGlobal<TViewSrc>& viewSrc)
    {
        using Type = std::remove_all_extents_t<TViewSrc>;
        auto extent = getExtents(viewDst);
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(&viewSrc.value)));

        auto view = alpaka::ViewPlainPtr<
            DevUniformCudaHipRt<TApi>,
            Type,
            alpaka::Dim<decltype(extent)>,
            alpaka::Idx<decltype(extent)>>(pMemAcc, alpaka::getDev(queue), extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDst>(viewDst), view, extent));
    }

    // from host to device
    template<typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        alpaka::DevGlobal<TViewDst>& viewDst,
        TViewSrc const& viewSrc)
    {
        using Type = std::remove_all_extents_t<TViewDst>;
        auto extent = getExtents(viewSrc);
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(&viewDst.value)));

        auto view = alpaka::ViewPlainPtr<
            DevUniformCudaHipRt<TApi>,
            Type,
            alpaka::Dim<decltype(extent)>,
            alpaka::Idx<decltype(extent)>>(pMemAcc, alpaka::getDev(queue), extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    // from device to host
    template<typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc, typename TExtent>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        TViewDst& viewDst,
        alpaka::DevGlobal<TViewSrc>& viewSrc,
        TExtent extent)
    {
        using Type = std::remove_all_extents_t<TViewSrc>;
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(&viewSrc.value)));

        auto view = alpaka::ViewPlainPtr<DevUniformCudaHipRt<TApi>, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            pMemAcc,
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDst>(viewDst), view, extent));
    }

    // from host to device
    template<typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc, typename TExtent>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        alpaka::DevGlobal<TViewDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent extent)
    {
        using Type = std::remove_all_extents_t<TViewDst>;
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(&viewDst.value)));

        auto view = alpaka::ViewPlainPtr<DevUniformCudaHipRt<TApi>, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            pMemAcc,
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }
} // namespace alpaka

#endif
