/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

// Base classes.
#include <alpaka/acc/AccGpuUniformedCudaHipRt.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/dev/DevUniformedCudaHipRt.hpp>
#include <alpaka/core/Hip.hpp>

#include <typeinfo>

namespace alpaka
{
    namespace kernel
    {
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelGpuUniformedCudaHipRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU HIP accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting HIP or HCC
        template<
            typename TDim,
            typename TIdx>
        class AccGpuUniformedCudaHipRt final :
            public acc::AccGpuUniformedCudaHipRt<TDim,TIdx>,
            public concepts::Implements<UnifiedAcc, AccGpuUniformedCudaHipRt<TDim, TIdx>>
            public concepts::Implements<ConceptAcc, AccGpuHipRt<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuUniformedCudaHipRt(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    AccGpuUniformedCudaHipRt<TDim,TIdx>(threadElemExtent)
            {}

        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuUniformedCudaHipRt(AccGpuUniformedCudaHipRt const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ AccGpuUniformedCudaHipRt(AccGpuUniformedCudaHipRt &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuUniformedCudaHipRt const &) -> AccGpuUniformedCudaHipRt & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuUniformedCudaHipRt &&) -> AccGpuUniformedCudaHipRt & = delete;
            //-----------------------------------------------------------------------------
            ~AccGpuUniformedCudaHipRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccGpuUniformedCudaHipRt<TDim, TIdx>>
            {
                using type = acc::AccGpuUniformedCudaHipRt<TDim, TIdx>;
            };
            
            //#############################################################################
            //! The GPU Hip accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccGpuUniformedCudaHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccGpuUniformedCudaHipRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccGpuCudaRt<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs && ... args)
                {
                    return
                        kernel::TaskKernelGpuUniformedCudaHipRt<
                            TDim,
                            TIdx,
                            TKernelFnObj,
                            TArgs...>(
                                workDiv,
                                kernelFnObj,
                                std::forward<TArgs>(args)...);
                }
            };
        }
    }
}

#endif
