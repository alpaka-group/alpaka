/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// ---------------------------------------------------------------------------
// Single-point environment detection for std::experimental::simd viability.
// This header is the ONLY place that decides whether the environment could
// support std::experimental::simd.
//
// Conditions (all must hold):
//  * Host side (no device compilation for CUDA/HIP/SYCL)
//  * Not compiled via NVCC/HIP frontends (mixed builds force fallback)
//  * Not using libc++ (historically missing <experimental/simd>)
//  * x86 family host
//  * <experimental/simd> header present and feature test macro defined
//
// Build control: CMake option alpaka_USE_STD_SIMD (macro ALPAKA_USE_STD_SIMD) can disable use even if available.
// ---------------------------------------------------------------------------

// Define Cmake variable if cmake is not used, make default ON
#ifndef ALPAKA_USE_STD_SIMD
#    define ALPAKA_USE_STD_SIMD 1
#endif
// Check if using <experimental/simd> is technically possible
#ifndef ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE
#    if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)                 \
        && !defined(__CUDACC__) && !defined(__NVCC__) && !defined(_LIBCPP_VERSION)                                    \
        && (defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64) || defined(__i386__)     \
            || defined(__i386) || defined(_M_IX86))
#        if defined(__has_include) && __has_include(<experimental/simd>)
#            include <experimental/simd>
#            if defined(__cpp_lib_experimental_parallel_simd)
#                define ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE 1
#            else
#                define ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE 0
#            endif
#        else
#            define ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE 0
#        endif
#    else
#        define ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE 0
#    endif
#endif


//   ALPAKA_USE_STD_SIMD && ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE
