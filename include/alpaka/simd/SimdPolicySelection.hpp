/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/acc/Tag.hpp>
#include <alpaka/core/Config.hpp>
// Single environment detection source
#include <alpaka/simd/SimdEnvironment.hpp>

#include <cstddef>
#include <type_traits>

namespace alpaka::simd
{
    //! ALPAKA has a "2-LEVEL" SIMD POLICY:
    //! Level 1: std::experimental::simd (GCC/Clang with libstdc++)
    //! Level 2: Fallback; actually an array-based auto-vectorization
    //!
    //! Policy selection rules:
    //! - std::experimental::simd: CPU-only builds with GCC+libstdc++ or compatible
    //! - Fallback: Mixed GPU+CPU builds, Clang+libc++, device compilation, or when std::experimental unavailable
    //!
    template<typename TAcc = void>
    struct SimdPolicySelection
    {
        static constexpr bool envHasStdExpSimd()
        {
            // If cmake variable has disabled std::experimental::simd, report false.
#ifdef ALPAKA_USE_STD_SIMD
            if constexpr(ALPAKA_USE_STD_SIMD == 0)
                return false;
#endif
            // if technically possible report true
            return (ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE != 0);
        }

        // Main determination code of SIMD policy.
        // 1. Check Cmake configuration for standard SIMD usage.  ALPAKA_USE_STD_SIMD is ON by default.
        // 2. Check compiler, stdlib and architecture. The macro is ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE.
        // 3. Check available accelerator backends
        static constexpr bool hasStdSimd = envHasStdExpSimd()
            && (std::is_void_v<TAcc>
                || ([]() constexpr {
                    if constexpr(std::is_void_v<TAcc>) {
                        return true;
                    } else {
                        return alpaka::accMatchesTags<
                            TAcc,
                            alpaka::TagCpuSerial,
                            alpaka::TagCpuThreads,
                            alpaka::TagCpuOmp2Blocks,
                            alpaka::TagCpuOmp2Threads,
                            alpaka::TagCpuTbbBlocks>;
                    }
                }()));

        static constexpr char const* name() noexcept
        {
            return hasStdSimd ? "std::experimental::simd" : "Fallback";
        }

        template<typename T>
        static constexpr std::size_t width() noexcept
        {
            // GPU accelerators use SIMD width 1 - parallelism through many threads
            if constexpr(!std::is_void_v<TAcc>)
            {
                if constexpr(alpaka::accMatchesTags<
                                 TAcc,
                                 alpaka::TagGpuCudaRt,
                                 alpaka::TagGpuHipRt,
                                 alpaka::TagGpuSyclIntel,
                                 alpaka::TagGenericSycl,
                                 alpaka::TagFpgaSyclIntel>)
                {
                    return 1u;
                }
            }

            // Base width derived from target ISA compile flags. This matches std::experimental::native_simd
            // when enabled (hasStdSimd==true). For the fallback implementation we may artificially raise
            // the minimum width for 32-bit element types to 8 to get better ILP/unrolling even on SSE2-only
            // CI builds compiled without -march=native. A previous mismatch between this function (returning 4)
            // and the fallback PortableSimd (using 8) caused out-of-bounds reads (segfault risk) when valid==W.
            std::size_t base =
#if defined(__AVX512F__)
                64u / sizeof(T);
#elif defined(__AVX2__)
                32u / sizeof(T);
#elif defined(__SSE2__)
                16u / sizeof(T);
#else
                // Portable lower bound
                4u;
#endif
            // Only apply the 8-lane minimum for the fallback backend (not for std::experimental) and only for 32-bit.
            if constexpr(sizeof(T) == 4)
            {
                if(!hasStdSimd && base < 8u)
                    return 8u;
            }
            return base;
        }
    };

    // Constexpr mirror for transitional use in C++ contexts
    constexpr bool kHasStdExpSimd = SimdPolicySelection<>::hasStdSimd;
} // namespace alpaka::simd
