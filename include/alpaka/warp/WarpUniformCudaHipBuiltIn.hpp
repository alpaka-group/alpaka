/* Copyright 2023 Sergei Bastrakov, David M. Rogers, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Config.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/warp/Traits.hpp"

#include <cstdint>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::warp
{
    //! The GPU CUDA/HIP warp.
    class WarpUniformCudaHipBuiltIn : public interface::Implements<ConceptWarp, WarpUniformCudaHipBuiltIn>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !ALPAKA_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !ALPAKA_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        template<>
        struct GetSize<WarpUniformCudaHipBuiltIn>
        {
            static __device__ auto getSize(warp::WarpUniformCudaHipBuiltIn const& /*warp*/) -> std::int32_t
            {
                return warpSize;
            }
        };

        template<>
        struct GetSizeCompileTime<WarpUniformCudaHipBuiltIn>
        {
            static constexpr __device__ auto getSizeCompileTime() -> std::int32_t
            {
#        if defined(__CUDA_ARCH__)
                // CUDA always has a warp size of 32
                return 32;
#        elif defined(__HIP_DEVICE_COMPILE__)
                // HIP/ROCm may have a wavefront of 32 or 64 depending on the target device
#            if defined(__GFX9__)
                // GCN 5.0 and CDNA GPUs have a wavefront size of 64
                return 64;
#            elif defined(__GFX10__) or defined(__GFX11__) or defined(__GFX12__)
                // RDNA GPUs have a wavefront size of 32
                return 32;
#            else
                // Unknown AMD GPU architecture
#                ifdef ALPAKA_DEFAULT_AMD_WAVEFRONT_SIZE
                return ALPAKA_DEFAULT_AMD_WAVEFRONT_SIZE
#                else
#                    error The current AMD GPU architucture is not supported by this version of alpaka. You can define a default wavefront size setting the preprocessor macro ALPAKA_DEFAULT_AMD_WAVEFRONT_SIZE
                return 0;
#                endif
#            endif
#        endif
                // Host compilation
                return 0;
            }
        };

        template<>
        struct GetSizeUpperLimit<WarpUniformCudaHipBuiltIn>
        {
            static constexpr __device__ auto getSizeUpperLimit() -> std::int32_t
            {
#        if defined(__CUDA_ARCH__)
                // CUDA always has a warp size of 32
                return 32;
#        elif defined(__HIP_DEVICE_COMPILE__)
                // HIP/ROCm may have a wavefront of 32 or 64 depending on the target device
#            if defined(__GFX9__)
                // GCN 5.0 and CDNA GPUs have a wavefront size of 64
                return 64;
#            elif defined(__GFX10__) or defined(__GFX11__) or defined(__GFX12__)
                // RDNA GPUs have a wavefront size of 32
                return 32;
#            else
                // Unknown AMD GPU architecture
#                ifdef ALPAKA_DEFAULT_AMD_WAVEFRONT_SIZE
                return ALPAKA_DEFAULT_AMD_WAVEFRONT_SIZE
#                else
#                    error The current AMD GPU architucture is not supported by this version of alpaka. You can define a default wavefront size setting the preprocessor macro ALPAKA_DEFAULT_AMD_WAVEFRONT_SIZE
                return 64;
#                endif
#            endif
#        endif
                // Host compilation
                return 64;
            }
        };

        template<>
        struct Activemask<WarpUniformCudaHipBuiltIn>
        {
            static __device__ auto activemask(warp::WarpUniformCudaHipBuiltIn const& /*warp*/)
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                -> std::uint32_t
#        else
                -> std::uint64_t
#        endif
            {
                return __activemask();
            }
        };

        template<>
        struct All<WarpUniformCudaHipBuiltIn>
        {
            static __device__ auto all(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                std::int32_t predicate) -> std::int32_t
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && HIP_VERSION >= 60'200'000)
                return __all_sync(activemask(warp), predicate);
#        else
                return __all(predicate);
#        endif
            }
        };

        template<>
        struct Any<WarpUniformCudaHipBuiltIn>
        {
            static __device__ auto any(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                std::int32_t predicate) -> std::int32_t
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && HIP_VERSION >= 60'200'000)
                return __any_sync(activemask(warp), predicate);
#        else
                return __any(predicate);
#        endif
            }
        };

        template<>
        struct Ballot<WarpUniformCudaHipBuiltIn>
        {
            static __device__ auto ballot(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                std::int32_t predicate)
            // return type is required by the compiler
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                -> std::uint32_t
#        else
                -> std::uint64_t
#        endif
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && HIP_VERSION >= 60'200'000)
                return __ballot_sync(activemask(warp), predicate);
            }
#        else
                return __ballot(predicate);
#        endif
        };

        template<>
        struct Shfl<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            static __device__ auto shfl(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                int srcLane,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && HIP_VERSION >= 60'200'000)
                return __shfl_sync(activemask(warp), val, srcLane, width);
#        else
                    return __shfl(val, srcLane, width);
#        endif
            }
        };

        template<>
        struct ShflUp<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            static __device__ auto shfl_up(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                std::uint32_t offset,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && HIP_VERSION >= 60'200'000)
                return __shfl_up_sync(activemask(warp), val, offset, width);
#        else
                    return __shfl_up(val, offset, width);
#        endif
            }
        };

        template<>
        struct ShflDown<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            static __device__ auto shfl_down(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                std::uint32_t offset,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && HIP_VERSION >= 60'200'000)
                return __shfl_down_sync(activemask(warp), val, offset, width);
#        else
                    return __shfl_down(val, offset, width);
#        endif
            }
        };

        template<>
        struct ShflXor<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            static __device__ auto shfl_xor(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                std::int32_t mask,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && HIP_VERSION >= 60'200'000)
                return __shfl_xor_sync(activemask(warp), val, mask, width);
#        else
                    return __shfl_xor(val, mask, width);
#        endif
            }
        };

    } // namespace trait
#    endif
} // namespace alpaka::warp

#endif
