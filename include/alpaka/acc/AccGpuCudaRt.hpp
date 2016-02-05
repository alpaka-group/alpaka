/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#ifndef __CUDACC__
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivCudaBuiltIn.hpp>    // WorkDivCudaBuiltIn
#include <alpaka/idx/gb/IdxGbCudaBuiltIn.hpp>       // IdxGbCudaBuiltIn
#include <alpaka/idx/bt/IdxBtCudaBuiltIn.hpp>       // IdxBtCudaBuiltIn
#include <alpaka/atomic/AtomicCudaBuiltIn.hpp>      // AtomicCudaBuiltIn
#include <alpaka/math/MathCudaBuiltIn.hpp>          // MathCudaBuiltIn
#include <alpaka/block/shared/dyn/BlockSharedMemDynCudaBuiltIn.hpp> // BlockSharedMemDynCudaBuiltIn
#include <alpaka/block/shared/st/BlockSharedMemStCudaBuiltIn.hpp>   // BlockSharedMemStCudaBuiltIn
#include <alpaka/block/sync/BlockSyncCudaBuiltIn.hpp>                   // BlockSyncCudaBuiltIn
#include <alpaka/rand/RandCuRand.hpp>               // RandCuRand

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                    // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                    // dev::traits::DevType
#include <alpaka/exec/Traits.hpp>                   // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>                   // size::traits::SizeType

// Implementation details.
#include <alpaka/dev/DevCudaRt.hpp>                 // dev::DevCudaRt
#include <alpaka/core/Cuda.hpp>                     // ALPAKA_CUDA_RT_CHECK

#include <boost/predef.h>                           // workarounds
#include <boost/align.hpp>                          // boost::aligned_alloc

#include <typeinfo>                                 // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecGpuCudaRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU CUDA accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting CUDA.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class AccGpuCudaRt final :
            public workdiv::WorkDivCudaBuiltIn<TDim, TSize>,
            public idx::gb::IdxGbCudaBuiltIn<TDim, TSize>,
            public idx::bt::IdxBtCudaBuiltIn<TDim, TSize>,
            public atomic::AtomicCudaBuiltIn,
            public math::MathCudaBuiltIn,
            public block::shared::dyn::BlockSharedMemDynCudaBuiltIn,
            public block::shared::st::BlockSharedMemStCudaBuiltIn,
            public block::sync::BlockSyncCudaBuiltIn
// This header is not currently supported by the clang native CUDA compiler.
#if !defined(__CUDA__)
            , public rand::RandCuRand
#endif
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AccGpuCudaRt(
                Vec<TDim, TSize> const & threadElemExtent) :
                    workdiv::WorkDivCudaBuiltIn<TDim, TSize>(threadElemExtent),
                    idx::gb::IdxGbCudaBuiltIn<TDim, TSize>(),
                    idx::bt::IdxBtCudaBuiltIn<TDim, TSize>(),
                    atomic::AtomicCudaBuiltIn(),
                    math::MathCudaBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynCudaBuiltIn(),
                    block::shared::st::BlockSharedMemStCudaBuiltIn(),
                    block::sync::BlockSyncCudaBuiltIn()
#if !defined(__CUDA__)
                    , rand::RandCuRand()
#endif
            {}

        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AccGpuCudaRt(AccGpuCudaRt const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AccGpuCudaRt(AccGpuCudaRt &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(AccGpuCudaRt const &) -> AccGpuCudaRt & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(AccGpuCudaRt &&) -> AccGpuCudaRt & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY /*virtual*/ ~AccGpuCudaRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = acc::AccGpuCudaRt<TDim, TSize>;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCudaRt const & dev)
                -> acc::AccDevProps<TDim, TSize>
                {
                    cudaDeviceProp cudaDevProp;
                    ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(
                        &cudaDevProp,
                        dev.m_iDevice));

                    return {
                        // m_multiProcessorCount
                        static_cast<TSize>(cudaDevProp.multiProcessorCount),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            Vec<dim::DimInt<3u>, TSize>(
                                static_cast<TSize>(cudaDevProp.maxGridSize[2]),
                                static_cast<TSize>(cudaDevProp.maxGridSize[1]),
                                static_cast<TSize>(cudaDevProp.maxGridSize[0]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TSize>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            Vec<dim::DimInt<3u>, TSize>(
                                static_cast<TSize>(cudaDevProp.maxThreadsDim[2]),
                                static_cast<TSize>(cudaDevProp.maxThreadsDim[1]),
                                static_cast<TSize>(cudaDevProp.maxThreadsDim[0]))),
                        // m_blockThreadCountMax
                        static_cast<TSize>(cudaDevProp.maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TSize>::max()};
                }
            };
            //#############################################################################
            //! The GPU CUDA accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccGpuCudaRt<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = dev::DevManCudaRt;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccGpuCudaRt<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
