/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/block/shared/st/Traits.hpp>

#include <type_traits>
#include <cstdint>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                //#############################################################################
                //! The GPU CUDA-HIP block shared memory allocator.
                class BlockSharedMemStUniformedCudaHipBuiltIn : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStUniformedCudaHipBuiltIn>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    BlockSharedMemStUniformedCudaHipBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    __device__ BlockSharedMemStUniformedCudaHipBuiltIn(BlockSharedMemStUniformedCudaHipBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    __device__ BlockSharedMemStUniformedCudaHipBuiltIn(BlockSharedMemStUniformedCudaHipBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    __device__ auto operator=(BlockSharedMemStUniformedCudaHipBuiltIn const &) -> BlockSharedMemStUniformedCudaHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    __device__ auto operator=(BlockSharedMemStUniformedCudaHipBuiltIn &&) -> BlockSharedMemStUniformedCudaHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    /*virtual*/ ~BlockSharedMemStUniformedCudaHipBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStUniformedCudaHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto allocVar(
                            block::shared::st::BlockSharedMemStUniformedCudaHipBuiltIn const &)
                        -> T &
                        {
                            __shared__ uint8_t shMem alignas(alignof(T)) [sizeof(T)];
                            return *(
                                reinterpret_cast<T*>( shMem ));
                        }
                    };
                    //#############################################################################
                    template<>
                    struct FreeMem<
                        BlockSharedMemStUniformedCudaHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto freeMem(
                            block::shared::st::BlockSharedMemStUniformedCudaHipBuiltIn const &)
                        -> void
                        {
                            // Nothing to do. CUDA/HIP block shared memory is automatically freed when all threads left the block.
                        }
                    };
                }
            }
        }
    }
}

#endif
