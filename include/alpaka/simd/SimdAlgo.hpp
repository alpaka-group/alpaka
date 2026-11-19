/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// for simdWidth + unified policy access
#include "SimdPolicySelection.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <cstdint> // uintptr_t for alignment check
#include <memory> // std::assume_aligned (C++20)
#include <type_traits>

namespace alpaka::simd
{
    template<typename T, typename TAcc>
    class PortableSimd;
} // namespace alpaka::simd

// Local restrict helper (mirrors definition in SimdCpu.hpp when that header
// is not yet included). Keeping it here avoids pulling in the whole CPU impl.
#ifndef ALPAKA_SIMD_RESTRICT
#    if defined(_MSC_VER)
#        define ALPAKA_SIMD_RESTRICT __restrict
#    else
#        define ALPAKA_SIMD_RESTRICT __restrict__
#    endif
#endif

// Small utility: guarded assume-aligned. We only apply on host side and when
// the pointer actually satisfies the alignment at runtime to avoid UB.
// Falls back to the original pointer otherwise. Kept extremely lightweight.
template<std::size_t Align, class T>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T* alpakaSimdAssumeAligned(T* p) noexcept
{
#if defined(__cpp_lib_assume_aligned) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)                  \
    && !defined(__SYCL_DEVICE_ONLY__)
    if((reinterpret_cast<std::uintptr_t>(p) & (Align - 1u)) == 0u)
        return std::assume_aligned<Align>(p);
#endif
    return p;
}

template<std::size_t Align, class T>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T const* alpakaSimdAssumeAligned(T const* p) noexcept
{
#if defined(__cpp_lib_assume_aligned) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)                  \
    && !defined(__SYCL_DEVICE_ONLY__)
    if((reinterpret_cast<std::uintptr_t>(p) & (Align - 1u)) == 0u)
        return std::assume_aligned<Align>(p);
#endif
    return p;
}

// Lightweight tail-hiding SIMD scheduling helper. If data-size is not multiple of simd-width there is a tail, this can
// be hided. Each logical thread processes a strided subset of fixed-size SIMD "packs"; the final pack may be partial.
// Kernels using this object contain no explicit tail code: partial packs are handled by guarded load/store.
// This adds onAcc::SimdAlgo concept while retaining the existing PortableSimd implementation.

namespace alpaka::simd
{
    template<typename Acc, typename T>
    struct SimdAlgo
    {
        // SIMD width (max lanes or max elements)
        std::size_t width;
        // Total data size
        std::size_t n;
        // ceil(n/width)
        std::size_t packCount;
        // thread id in grid (1D)
        std::size_t tid;
        // threads in grid (1D)
        std::size_t nThreads;

        ALPAKA_FN_ACC SimdAlgo(Acc const& acc, std::size_t nElems)
            : width(alpaka::simd::SimdPolicySelection<Acc>::template width<T>())
            , n(nElems)
            , packCount((nElems + width - 1u) / width)
            , tid(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0])
            , nThreads(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0])
        {
        }

        template<typename F>
        ALPAKA_FN_ACC void forEach(F&& f) const
        {
            // Grid-stride loop: each thread processes multiple packs
            // Thread 0 handles packs 0, nThreads, 2*nThreads...; Thread 1 handles packs 1, nThreads+1, 2*nThreads+1...
            // This ensures all packs [0, packCount) are covered even when nThreads < packCount.
            for(std::size_t pack = tid; pack < packCount; pack += nThreads)
            {
                std::size_t base = pack * width;
                std::size_t remaining = n - base;
                // Lanes to process in this pack
                std::size_t valid = remaining >= width ? width : remaining;
                f(base, valid);
            }
        }
    };

    // Convenience helpers for (partial) pack load/store without exposing tail logic to kernels.
    template<typename T, typename Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void loadSimd(
        T const* ALPAKA_SIMD_RESTRICT src,
        std::size_t valid,
        PortableSimd<T, Acc>& dst) noexcept
    {
        auto const W = alpaka::simd::SimdPolicySelection<Acc>::template width<T>();
        if(valid == W)
        {
            // Try an aligned fast path first (host only). Use 32-byte default; upgrade
            // to 64 for wide (>=16 * 4B) packs – avoids unnecessary vmovdqu.
            constexpr std::size_t kElemBytes = sizeof(T);
            constexpr std::size_t kPreferredAlign = (kElemBytes * 16u >= 64u) ? 64u : 32u;
            auto alignedPtr = alpakaSimdAssumeAligned<kPreferredAlign>(src);
            dst.load(alignedPtr);
            return;
        }
        // Tail: use dynamic-size scratch buffer based on actual SIMD width
        // This avoids hard-coded architecture assumptions and uses only the memory actually needed
        T laneBuf[W];
        for(std::size_t i = 0; i < W; ++i)
            laneBuf[i] = (i < valid) ? src[i] : T{0};
        // PortableSimd ignores extra lanes beyond its native width (only first W used)
        dst.load(laneBuf);
    }

    template<typename T, typename Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void storeSimd(
        PortableSimd<T, Acc> const& src,
        std::size_t valid,
        T* ALPAKA_SIMD_RESTRICT dst) noexcept
    {
        auto const W = alpaka::simd::SimdPolicySelection<Acc>::template width<T>();
        if(valid == W)
        {
            constexpr std::size_t kElemBytes = sizeof(T);
            constexpr std::size_t kPreferredAlign = (kElemBytes * 16u >= 64u) ? 64u : 32u;
            auto alignedPtr = alpakaSimdAssumeAligned<kPreferredAlign>(dst);
            src.store(alignedPtr);
            return;
        }
        // Tail: use dynamic-size scratch buffer based on actual SIMD width
        T laneBuf[W];
        // Store whole (native) pack then copy only the valid portion.
        src.store(laneBuf);
        for(std::size_t i = 0; i < valid; ++i)
            dst[i] = laneBuf[i];
    }
} // namespace alpaka::simd
