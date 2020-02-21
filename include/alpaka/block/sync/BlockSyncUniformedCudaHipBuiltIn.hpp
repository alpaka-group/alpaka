/* Copyright 2019 Benjamin Worpitz
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

#include <alpaka/block/sync/Traits.hpp>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The GPU CUDA-HIP block synchronization.
            class BlockSyncUniformedCudaHipBuiltIn : public concepts::Implements<ConceptBlockSync, BlockSyncUniformedCudaHipBuiltIn>
            {
            public:
                //-----------------------------------------------------------------------------
                BlockSyncUniformedCudaHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                __device__ BlockSyncUniformedCudaHipBuiltIn(BlockSyncUniformedCudaHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                __device__ BlockSyncUniformedCudaHipBuiltIn(BlockSyncUniformedCudaHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(BlockSyncUniformedCudaHipBuiltIn const &) -> BlockSyncUniformedCudaHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(BlockSyncUniformedCudaHipBuiltIn &&) -> BlockSyncUniformedCudaHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncUniformedCudaHipBuiltIn() = default;
            };

            namespace traits
            {
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncUniformedCudaHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreads(
                        block::sync::BlockSyncUniformedCudaHipBuiltIn const & /*blockSync*/)
                    -> void
                    {
                        __syncthreads();
                    }
                };

                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::Count,
                    BlockSyncUniformedCudaHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncUniformedCudaHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
#if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__==0 && (BOOST_COMP_HCC || BOOST_COMP_HIP)
                        // workaround for unsupported syncthreads_* operation on HIP(HCC)
                        __shared__ int tmp;
                        __syncthreads();
                        if(threadIdx.x==0)
                            tmp=0;
                        __syncthreads();
                        if(predicate)
                            atomicAdd(&tmp, 1);
                        __syncthreads();

                        return tmp;
#else
                        return __syncthreads_count(predicate);
#endif
                    }
                };

                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalAnd,
                    BlockSyncUniformedCudaHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncUniformedCudaHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
#if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__==0 && (BOOST_COMP_HCC || BOOST_COMP_HIP)
                        // workaround for unsupported syncthreads_* operation on HIP(HCC)
                        __shared__ int tmp;
                        __syncthreads();
                        if(threadIdx.x==0)
                            tmp=1;
                        __syncthreads();
                        if(!predicate)
                            atomicAnd(&tmp, 0);
                        __syncthreads();

                        return tmp;
#else
                        return __syncthreads_and(predicate);
#endif
                    }
                };

                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalOr,
                    BlockSyncUniformedCudaHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncUniformedCudaHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
#if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__==0 && (BOOST_COMP_HCC || BOOST_COMP_HIP)
                        // workaround for unsupported syncthreads_* operation on HIP(HCC)
                        __shared__ int tmp;
                        __syncthreads();
                        if(threadIdx.x==0)
                            tmp=0;
                        __syncthreads();
                        if(predicate)
                            atomicOr(&tmp, 1);
                        __syncthreads();

                        return tmp;
#else
                        return __syncthreads_or(predicate);
#endif
                    }
                };
            }
        }
    }
}

#endif
