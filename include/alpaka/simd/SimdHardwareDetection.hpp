/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/core/Common.hpp>

#include <cstddef>

namespace alpaka::simd::detail
{
    //! SIMD instruction set detection
    //! This file provides compile-time detection for CPU SIMD capabilities

    // Named constants for SIMD register sizes to avoid magic numbers
    namespace constants
    {
        constexpr std::size_t AVX512_REGISTER_BITS = 512;
        constexpr std::size_t AVX2_REGISTER_BITS = 256;
        constexpr std::size_t SSE2_REGISTER_BITS = 128;
        constexpr std::size_t BITS_PER_BYTE = 8;
        constexpr std::size_t FALLBACK_REGISTER_BYTES = sizeof(void*);
    } // namespace constants

    // SIMD instruction set detection
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr bool hasAVX512F() noexcept
    {
#if defined(__AVX512F__)
        return true;
#else
        return false;
#endif
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr bool hasAVX2() noexcept
    {
#if defined(__AVX2__)
        return true;
#else
        return false;
#endif
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr bool hasSSE2() noexcept
    {
#if defined(__SSE2__)
        return true;
#else
        return false;
#endif
    }

} // namespace alpaka::simd::detail
