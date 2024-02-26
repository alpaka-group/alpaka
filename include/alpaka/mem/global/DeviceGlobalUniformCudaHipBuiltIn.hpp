/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/mem/global/Traits.hpp"
#include "alpaka/queue/cuda_hip/QueueUniformCudaHipRt.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{

    namespace detail
    {
        template<typename T>
        struct DevGlobalTrait<TagGpuCudaRt, T>
        {
            // CUDA implementation
            using Type = detail::DevGlobalImplGeneric<TagGpuCudaRt, T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagGpuHipRt, T>
        {
            // HIP/ROCm implementation
            using Type = detail::DevGlobalImplGeneric<TagGpuHipRt, T>;
        };
    } // namespace detail

    // from device to host
    template<typename TTag, typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        TViewDst& viewDst,
        alpaka::detail::DevGlobalImplGeneric<TTag, TViewSrc>& viewSrc)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TViewSrc>>;
        using TypeExt = std::remove_const_t<TViewSrc>;
        auto extent = getExtents(viewDst);
        TypeExt* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getSymbolAddress(
            reinterpret_cast<void**>(&pMemAcc),
            *(const_cast<TypeExt*>(&viewSrc))));

        auto view = alpaka::ViewPlainPtr<
            DevUniformCudaHipRt<TApi>,
            Type,
            alpaka::Dim<decltype(extent)>,
            alpaka::Idx<decltype(extent)>>(reinterpret_cast<Type*>(pMemAcc), alpaka::getDev(queue), extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDst>(viewDst), view, extent));
    }

    // from host to device
    template<typename TTag, typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        alpaka::detail::DevGlobalImplGeneric<TTag, TViewDst>& viewDst,
        TViewSrc const& viewSrc)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TViewDst>>;
        using TypeExt = std::remove_const_t<TViewDst>;
        auto extent = getExtents(viewSrc);
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getSymbolAddress(
            reinterpret_cast<void**>(&pMemAcc),
		    *(const_cast<TypeExt*>(&viewDst))));

        auto view = alpaka::ViewPlainPtr<
            DevUniformCudaHipRt<TApi>,
            Type,
            alpaka::Dim<decltype(extent)>,
            alpaka::Idx<decltype(extent)>>(reinterpret_cast<Type*>(pMemAcc), alpaka::getDev(queue), extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    // from device to host
    template<typename TTag, typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc, typename TExtent>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        TViewDst& viewDst,
        alpaka::detail::DevGlobalImplGeneric<TTag, TViewSrc>& viewSrc,
        TExtent extent)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TViewSrc>>;
		using TypeExt = std::remove_const_t<TViewSrc>;
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getSymbolAddress(
            reinterpret_cast<void**>(&pMemAcc),
		    *(const_cast<TypeExt*>(&viewSrc))));

        auto view = alpaka::ViewPlainPtr<DevUniformCudaHipRt<TApi>, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(pMemAcc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDst>(viewDst), view, extent));
    }

    // from host to device
    template<typename TTag, typename TApi, bool TBlocking, typename TViewDst, typename TViewSrc, typename TExtent>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        alpaka::detail::DevGlobalImplGeneric<TTag, TViewDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent extent)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TViewDst>>;
		using TypeExt = std::remove_const_t<TViewDst>;
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getSymbolAddress(
            reinterpret_cast<void**>(&pMemAcc),
		    *(const_cast<TypeExt*>(&viewDst))));

        auto view = alpaka::ViewPlainPtr<DevUniformCudaHipRt<TApi>, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(pMemAcc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }
} // namespace alpaka

#endif
