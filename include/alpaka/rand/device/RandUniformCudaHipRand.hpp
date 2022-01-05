/* Copyright 2022 Benjamin Worpitz, René Widera, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/DeviceOnly.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/rand/RandUniformCudaHipRand.hpp>

// Backend specific imports.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>

#        include <curand_kernel.h>

#    elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        include <alpaka/core/Hip.hpp>

#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wduplicate-decl-specifier"

#        include <hiprand_kernel.h>

#        pragma clang diagnostic pop
#    endif

#    include <type_traits>

namespace alpaka
{
    namespace rand
    {
        namespace engine
        {
            namespace uniform_cuda_hip
            {
                //! The CUDA/HIP Xor random number generator engine.
                class Xor
                {
                public:
                    // After calling this constructor the instance is not valid initialized and
                    // need to be overwritten with a valid object
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_FN_HOST_ACC Xor() : state(curandStateXORWOW_t{})
#    else
                    ALPAKA_FN_HOST_ACC Xor() : state(hiprandStateXORWOW_t{})
#    endif
                    {
                    }

                    __device__ Xor(
                        std::uint32_t const& seed,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0)
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        curand_init(seed, subsequence, offset, &state);
#    else
                        hiprand_init(seed, subsequence, offset, &state);
#    endif
                    }

                public:
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    curandStateXORWOW_t state;
#    else
                    hiprandStateXORWOW_t state;
#    endif

                    // STL UniformRandomBitGenerator concept. This is not strictly necessary as the distributions
                    // contained in this file are aware of the API specifics of the CUDA/HIP XORWOW engine and STL
                    // distributions might not work on the device, but it servers a compatibility bridge to other
                    // potentially compatible alpaka distributions.
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    using result_type = decltype(curand(&state));
#    else
                    using result_type = decltype(hiprand(&state));
#    endif
                    ALPAKA_FN_HOST_ACC constexpr static result_type min()
                    {
                        return std::numeric_limits<result_type>::min();
                    }
                    ALPAKA_FN_HOST_ACC constexpr static result_type max()
                    {
                        return std::numeric_limits<result_type>::max();
                    }
                    __device__ result_type operator()()
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        return curand(&state);
#    else
                        return hiprand(&state);
#    endif
                    }
                };
            } // namespace uniform_cuda_hip
        } // namespace engine

        namespace distribution
        {
            namespace uniform_cuda_hip
            {
                //! The CUDA/HIP random number floating point normal distribution.
                template<typename T>
                class NormalReal;

                //! The CUDA/HIP random number float normal distribution.
                template<>
                class NormalReal<float>
                {
                public:
                    template<typename TEngine>
                    __device__ auto operator()(TEngine& engine) -> float
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        return curand_normal(&engine.state);
#    else
                        return hiprand_normal(&engine.state);
#    endif
                    }
                };

                //! The CUDA/HIP random number float normal distribution.
                template<>
                class NormalReal<double>
                {
                public:
                    template<typename TEngine>
                    __device__ auto operator()(TEngine& engine) -> double
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        return curand_normal_double(&engine.state);
#    else
                        return hiprand_normal_double(&engine.state);
#    endif
                    }
                };

                //! The CUDA/HIP random number floating point uniform distribution.
                template<typename T>
                class UniformReal;

                //! The CUDA/HIP random number float uniform distribution.
                template<>
                class UniformReal<float>
                {
                public:
                    template<typename TEngine>
                    __device__ auto operator()(TEngine& engine) -> float
                    {
                        // (0.f, 1.0f]
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        float const fUniformRand(curand_uniform(&engine.state));
#    else
                        float const fUniformRand(hiprand_uniform(&engine.state));
#    endif
                        // NOTE: (1.0f - curand_uniform) does not work, because curand_uniform seems to return
                        // denormalized floats around 0.f. [0.f, 1.0f)
                        return fUniformRand * static_cast<float>(fUniformRand != 1.0f);
                    }
                };

                //! The CUDA/HIP random number float uniform distribution.
                template<>
                class UniformReal<double>
                {
                public:
                    template<typename TEngine>
                    __device__ auto operator()(TEngine& engine) -> double
                    {
                        // (0.f, 1.0f]
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        double const fUniformRand(curand_uniform_double(&engine.state));
#    else
                        double const fUniformRand(hiprand_uniform_double(&engine.state));
#    endif
                        // NOTE: (1.0f - curand_uniform_double) does not work, because curand_uniform_double seems to
                        // return denormalized floats around 0.f. [0.f, 1.0f)
                        return fUniformRand * static_cast<double>(fUniformRand != 1.0);
                    }
                };

                //! The CUDA/HIP random number integer uniform distribution.
                template<typename T>
                class UniformUint;

                //! The CUDA/HIP random number unsigned integer uniform distribution.
                template<>
                class UniformUint<unsigned int>
                {
                public:
                    template<typename TEngine>
                    __device__ auto operator()(TEngine& engine) -> unsigned int
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        return curand(&engine.state);
#    else
                        return hiprand(&engine.state);
#    endif
                    }
                };
            } // namespace uniform_cuda_hip
        } // namespace distribution

        namespace distribution
        {
            namespace traits
            {
                //! The CUDA/HIP random number float normal distribution get trait specialization.
                template<typename T>
                struct CreateNormalReal<RandUniformCudaHipRand, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    __device__ static auto createNormalReal(RandUniformCudaHipRand const& /*rand*/)
                        -> rand::distribution::uniform_cuda_hip::NormalReal<T>
                    {
                        return rand::distribution::uniform_cuda_hip::NormalReal<T>();
                    }
                };

                //! The CUDA/HIP random number float uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformReal<RandUniformCudaHipRand, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    __device__ static auto createUniformReal(RandUniformCudaHipRand const& /*rand*/)
                        -> rand::distribution::uniform_cuda_hip::UniformReal<T>
                    {
                        return rand::distribution::uniform_cuda_hip::UniformReal<T>();
                    }
                };

                //! The CUDA/HIP random number integer uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformUint<RandUniformCudaHipRand, T, std::enable_if_t<std::is_integral<T>::value>>
                {
                    __device__ static auto createUniformUint(RandUniformCudaHipRand const& /*rand*/)
                        -> rand::distribution::uniform_cuda_hip::UniformUint<T>
                    {
                        return rand::distribution::uniform_cuda_hip::UniformUint<T>();
                    }
                };
            } // namespace traits
        } // namespace distribution

        namespace engine
        {
            namespace traits
            {
                //! The CUDA/HIP random number default generator get trait specialization.
                template<>
                struct CreateDefault<RandUniformCudaHipRand>
                {
                    __device__ static auto createDefault(
                        RandUniformCudaHipRand const& /*rand*/,
                        std::uint32_t const& seed = 0,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0) -> rand::engine::uniform_cuda_hip::Xor
                    {
                        return rand::engine::uniform_cuda_hip::Xor(seed, subsequence, offset);
                    }
                };
            } // namespace traits
        } // namespace engine
    } // namespace rand
} // namespace alpaka

#endif
