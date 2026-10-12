/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// Alpaka Simd Implementation
// =====  Two-Level SIMD Policy =====
// Level 1: std::experimental::simd (GCC/Clang + libstdc++ + CPU-only)
// Level 2: Array-based fallback (CUDA+CPU, Clang+libc++, or forced)
//
// API Options:
// 1. Explicit Simd types and operations
// 2. SIMD-friendly algorithm SimdAlgo::ForEach
//
// CMake option: alpaka_USE_STD_SIMD (default ON) controls enabling std::experimental::simd when supported
#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Vectorize.hpp"

#include <cstdint>
#include <type_traits>

// Include SIMD environment and policy selection (needed for traits and implementations)
#include "alpaka/simd/SimdEnvironment.hpp"
#include "alpaka/simd/SimdHardwareDetection.hpp"
#include "alpaka/simd/SimdPolicySelection.hpp"

// Unified CPU SIMD implementation (it internally selects std::experimental or fallback)
// This is safe under NVCC because the implementation forces fallback when compiling for CUDA.
#include "alpaka/simd/detail/SimdCpu.hpp"

// CPU fallback implementations (inherit from CPU serial)
#include "alpaka/simd/detail/SimdCpuFallback.hpp"

// Include GPU implementations
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include "alpaka/simd/detail/SimdGpu.hpp"
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
#    include "alpaka/simd/detail/SimdSycl.hpp"
#endif

// Include SIMD algorithm helpers
#include "alpaka/simd/SimdAlgo.hpp"

// Include SIMD information and debug utilities
#include "alpaka/simd/SimdInfo.hpp"

namespace alpaka::simd
{
    //! Type alias for easier usage
    template<typename T, typename TAcc>
    using Simd = PortableSimd<T, TAcc>;

    // ---------------------------------------------------------------------
    // Lightweight accelerator classification helpers. GPU must be checked before multi-thread
    // because GPU accelerators satisfy isMultiThreadAcc.
    // ---------------------------------------------------------------------

    template<typename Acc>
    constexpr bool isGpuAcc()
    {
        return alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt, alpaka::TagGpuHipRt>;
    }

    template<typename Acc>
    constexpr bool isCpuSerialAcc()
    {
        return alpaka::accMatchesTags<Acc, alpaka::TagCpuSerial>;
    }

    template<typename Acc>
    constexpr bool isCpuMultiThreadAcc()
    {
        // Exclude GPU so callers can rely on mutual exclusivity if desired.
        return !isGpuAcc<Acc>() && alpaka::isMultiThreadAcc<Acc>;
    }

    // SIMD Traits and Type Aliases
    // Additional type aliases for SIMD vector types
    template<typename T, typename TAcc>
    using SimdType = Simd<T, TAcc>;

    // Width accessor
    template<typename T, typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::size_t simdWidth() noexcept
    {
        // Delegate width selection to unified capability (single source of truth).
        return SimdPolicySelection<TAcc>::template width<T>();
    }

    // Policy name utility
    template<typename T, typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr char const* policyName() noexcept
    {
        return SimdPolicySelection<TAcc>::name();
    }

} // namespace alpaka::simd
