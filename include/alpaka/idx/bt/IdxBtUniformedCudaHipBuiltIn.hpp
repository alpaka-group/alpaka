/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
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

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The CUDA-HIP accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxBtUniformedCudaHipBuiltIn : public concepts::Implements<ConceptIdxBt, IdxBtUniformedCudaHipBuiltIn<TDim, TIdx>>
            {
            public:
                //-----------------------------------------------------------------------------
                IdxBtUniformedCudaHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                __device__ IdxBtUniformedCudaHipBuiltIn(IdxBtUniformedCudaHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                __device__ IdxBtUniformedCudaHipBuiltIn(IdxBtUniformedCudaHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(IdxBtUniformedCudaHipBuiltIn const & ) -> IdxBtUniformedCudaHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(IdxBtUniformedCudaHipBuiltIn &&) -> IdxBtUniformedCudaHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtUniformedCudaHipBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA-HIP accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtUniformedCudaHipBuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA-HIP accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtUniformedCudaHipBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                __device__ static auto getIdx(
                    idx::bt::IdxBtUniformedCudaHipBuiltIn<TDim, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(idx);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    return vec::cast<TIdx>(offset::getOffsetVecEnd<TDim>(threadIdx));
#else
                    return offset::getOffsetVecEnd<TDim>(
                        vec::Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                            static_cast<TIdx>(hipThreadIdx_z),
                            static_cast<TIdx>(hipThreadIdx_y),
                            static_cast<TIdx>(hipThreadIdx_x)));
#endif
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA-HIP accelerator block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtUniformedCudaHipBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
