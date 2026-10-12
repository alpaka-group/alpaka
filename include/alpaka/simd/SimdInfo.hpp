/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

//! SIMD System Information and Diagnostic Utilities
//!
//! This file provides functions for debugging and information retrieval/logging purposes
//! related to SIMD capabilities, compiler environment, and accelerator backend setup.
//!
//! Usage: Include this header and call printSimdSystemInfo() in examples to get
//! comprehensive diagnostic output for troubleshooting SIMD performance or policy selection.

#pragma once

#include <alpaka/simd/SimdHardwareDetection.hpp>
#include <alpaka/simd/SimdPolicySelection.hpp>

#include <string>
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)
#    include <iostream>
#endif

// Avoid namespace conflicts with std::experimental
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)
using std::cout;
using std::endl;
#endif
using std::string;

namespace alpaka::simd
{
    //! Get compiler information as a string
    inline std::string getCompilerInfo()
    {
        std::string compiler = "Unknown";

#if ALPAKA_COMP_GNUC
        compiler = "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif ALPAKA_COMP_CLANG
        compiler = "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif ALPAKA_COMP_MSVC
        compiler = "MSVC " + std::to_string(_MSC_VER);
#endif

        return compiler;
    }

    //! Get standard library information
    inline std::string getStdLibInfo()
    {
        std::string stdlib = "Unknown";

#if defined(_GLIBCXX_RELEASE)
        stdlib = "libstdc++ " + std::to_string(_GLIBCXX_RELEASE);
#elif defined(_LIBCPP_VERSION)
        stdlib = "libc++ " + std::to_string(_LIBCPP_VERSION);
#elif ALPAKA_COMP_MSVC
        stdlib = "MSVC STL";
#endif

        return stdlib;
    }

    //! Get SIMD instruction set information
    inline std::string getSimdInstructionSet()
    {
        if constexpr(detail::hasAVX512F())
            return "AVX-512";
        else if constexpr(detail::hasAVX2())
            return "AVX2";
        else if constexpr(detail::hasSSE2())
            return "SSE2";
        else
            return "None";
    }

    //! Get SIMD policy type (std::experimental::simd vs fallback implementation)
    inline std::string getSimdPolicy()
    {
        // Device compilation contexts keep explicit labeling; host uses unified capability name.
#if defined(__CUDA_ARCH__)
        return "Device fallback (CUDA GPU)";
#elif defined(__SYCL_DEVICE_ONLY__)
        return "Device fallback (SYCL)";
#elif defined(__HIP_DEVICE_COMPILE__)
        return "Device fallback (HIP)";
#else
        return alpaka::simd::SimdPolicySelection<>::name();
#endif
    }

    //! Get accelerator backend setup status
    inline std::string getAcceleratorBackendStatus()
    {
        bool hasGPU = false;
        std::string cpuBackends = "";

        // Check for any GPU backend
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_SYCL_ENABLED)
        hasGPU = true;
#endif

        // Collect CPU backend types
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
        cpuBackends += "OMP+";
#endif
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
        cpuBackends += "OMP2Blocks+";
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
        cpuBackends += "Threads+";
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
        cpuBackends += "TBB+";
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
        cpuBackends += "Serial+";
#endif

        // Remove trailing '+'
        if(!cpuBackends.empty())
            cpuBackends.pop_back();

        // Format result
        if(hasGPU && !cpuBackends.empty())
            return "GPU + CPU (" + cpuBackends + ")";
        else if(hasGPU)
            return "GPU only";
        else if(!cpuBackends.empty())
            return "CPU only (" + cpuBackends + ")";
        else
            return "No backends detected";
    }

    //! Get detailed compiler environment for debugging
    inline std::string getCompilerEnvironment()
    {
        std::string env = "";

        // Compiler detection
#ifdef __NVCC__
        env += "NVCC compiler | ";
#endif
#ifdef __CUDACC__
        env += "CUDA compilation mode | ";
#endif
#ifdef __HIP__
        env += "HIP compiler | ";
#endif
#ifdef __SYCL_DEVICE_ONLY__
        env += "SYCL device compilation | ";
#endif

        // Backend detection
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        env += "CUDA backend enabled | ";
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        env += "HIP backend enabled | ";
#endif
#ifdef ALPAKA_ACC_GPU_SYCL_ENABLED
        env += "SYCL backend enabled | ";
#endif

        // Standard library detection
#ifdef _LIBCPP_VERSION
        env += "libc++ stdlib | ";
#elif defined(__GLIBCXX__)
        env += "libstdc++ stdlib | ";
#elif defined(_MSC_VER)
        env += "MSVC stdlib | ";
#endif

        // ISA instruction sets (hierarchical - show highest available)
#ifdef __AVX512F__
        env += "AVX-512 instructions | ";
#elif defined(__AVX2__)
        env += "AVX2 instructions | ";
#elif defined(__AVX__)
        env += "AVX instructions | ";
#elif defined(__SSE4_2__)
        env += "SSE4.2 instructions | ";
#elif defined(__SSE4_1__)
        env += "SSE4.1 instructions | ";
#elif defined(__SSSE3__)
        env += "SSSE3 instructions | ";
#elif defined(__SSE3__)
        env += "SSE3 instructions | ";
#elif defined(__SSE2__)
        env += "SSE2 instructions | ";
#elif defined(__SSE__)
        env += "SSE instructions | ";
#else
        env += "No SIMD instructions | ";
#endif

        // Optimization and debug info
#ifdef NDEBUG
        env += "Release build";
#else
        env += "Debug build";
#endif

        return env.empty() ? "Standard C++ environment" : env;
    }

    //! Print comprehensive SIMD system information
    inline void printSimdSystemInfo()
    {
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)
        std::cout << "\n=== SIMD System Information ===" << std::endl;
        std::cout << "Compiler: " << getCompilerInfo() << std::endl;
        std::cout << "Standard Library: " << getStdLibInfo() << std::endl;
        std::cout << "SIMD Instructions: " << getSimdInstructionSet() << std::endl;
        std::cout << "SIMD Policy: " << getSimdPolicy() << std::endl;
        std::cout << "Accelerator Setup: " << getAcceleratorBackendStatus() << std::endl;
        std::cout << "Compiler Environment: " << getCompilerEnvironment() << std::endl;
        std::cout << "================================" << std::endl;
#endif
    }

} // namespace alpaka::simd
