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
                class BlockSharedMemStCudaHipBuiltIn : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStCudaHipBuiltIn>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    BlockSharedMemStCudaHipBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    __device__ BlockSharedMemStCudaHipBuiltIn(BlockSharedMemStCudaHipBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    __device__ BlockSharedMemStCudaHipBuiltIn(BlockSharedMemStCudaHipBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    __device__ auto operator=(BlockSharedMemStCudaHipBuiltIn const &) -> BlockSharedMemStCudaHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    __device__ auto operator=(BlockSharedMemStCudaHipBuiltIn &&) -> BlockSharedMemStCudaHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    /*virtual*/ ~BlockSharedMemStCudaHipBuiltIn() = default;
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
                        BlockSharedMemStCudaHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto allocVar(
                            block::shared::st::BlockSharedMemStCudaHipBuiltIn const &)
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
                        BlockSharedMemStCudaHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto freeMem(
                            block::shared::st::BlockSharedMemStCudaHipBuiltIn const &)
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
