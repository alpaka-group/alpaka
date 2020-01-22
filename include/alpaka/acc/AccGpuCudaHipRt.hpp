/* Copyright 2019 Benjamin Worpitz, René Widera
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

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#endif

// Base classes.
#include <alpaka/workdiv/WorkDivCudaHipBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbCudaHipBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtCudaHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicCudaHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathCudaHipBuiltIn.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynCudaHipBuiltIn.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStCudaHipBuiltIn.hpp>
#include <alpaka/block/sync/BlockSyncCudaHipBuiltIn.hpp>
#include <alpaka/rand/RandCudaHipRand.hpp>
#include <alpaka/time/TimeCudaHipBuiltIn.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/dev/DevCudaHipRt.hpp>

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
        class TaskKernelGpuCudaHipRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU CUDA accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting CUDA.
        template<
            typename TDim,
            typename TIdx>
        class AccGpuCudaHipRt :
            public workdiv::WorkDivCudaHipBuiltIn<TDim, TIdx>,
            public idx::gb::IdxGbCudaHipBuiltIn<TDim, TIdx>,
            public idx::bt::IdxBtCudaHipBuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicCudaHipBuiltIn, // grid atomics
                atomic::AtomicCudaHipBuiltIn, // block atomics
                atomic::AtomicCudaHipBuiltIn  // thread atomics
            >,
            public math::MathCudaHipBuiltIn,
            public block::shared::dyn::BlockSharedMemDynCudaHipBuiltIn,
            public block::shared::st::BlockSharedMemStCudaHipBuiltIn,
            public block::sync::BlockSyncCudaHipBuiltIn,
            public rand::RandCudaHipRand,
            public time::TimeCudaHipBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuCudaHipRt(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    workdiv::WorkDivCudaHipBuiltIn<TDim, TIdx>(threadElemExtent),
                    idx::gb::IdxGbCudaHipBuiltIn<TDim, TIdx>(),
                    idx::bt::IdxBtCudaHipBuiltIn<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicCudaHipBuiltIn, // atomics between grids
                        atomic::AtomicCudaHipBuiltIn, // atomics between blocks
                        atomic::AtomicCudaHipBuiltIn  // atomics between threads
                    >(),
                    math::MathCudaHipBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynCudaHipBuiltIn(),
                    block::shared::st::BlockSharedMemStCudaHipBuiltIn(),
                    block::sync::BlockSyncCudaHipBuiltIn(),
                    rand::RandCudaHipRand(),
                    time::TimeCudaHipBuiltIn()
            {}

        public:

            //using baseType = AccCudaHip<TDim,TIdx>;
            
            //-----------------------------------------------------------------------------
            __device__ AccGpuCudaHipRt(AccGpuCudaHipRt const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ AccGpuCudaHipRt(AccGpuCudaHipRt &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuCudaHipRt const &) -> AccGpuCudaHipRt & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuCudaHipRt &&) -> AccGpuCudaHipRt & = delete;
            //-----------------------------------------------------------------------------
            ~AccGpuCudaHipRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccGpuCudaHipRt<TDim, TIdx>>
            {
                using type = acc::AccGpuCudaHipRt<TDim, TIdx>;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccGpuCudaHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCudaHipRt const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    // Reading only the necessary attributes with cudaDeviceGetAttribute is faster than reading all with cudaHipGetDeviceProperties
                    // https://devblogs.nvidia.com/cudaHip-pro-tip-the-fast-way-to-query-device-properties/
                    int multiProcessorCount = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &multiProcessorCount,
                        cudaDevAttrMultiProcessorCount,
                        dev.m_iDevice));

                    int maxGridSize[3] = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[0],
                        cudaDevAttrMaxGridDimX,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[1],
                        cudaDevAttrMaxGridDimY,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[2],
                        cudaDevAttrMaxGridDimZ,
                        dev.m_iDevice));

                    int maxBlockDim[3] = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[0],
                        cudaDevAttrMaxBlockDimX,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[1],
                        cudaDevAttrMaxBlockDimY,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[2],
                        cudaDevAttrMaxBlockDimZ,
                        dev.m_iDevice));

                    int maxThreadsPerBlock = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxThreadsPerBlock,
                        cudaDevAttrMaxThreadsPerBlock,
                        dev.m_iDevice));

                    return {
                        // m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(multiProcessorCount),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(maxGridSize[2u]),
                                alpaka::core::clipCast<TIdx>(maxGridSize[1u]),
                                alpaka::core::clipCast<TIdx>(maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(maxBlockDim[2u]),
                                alpaka::core::clipCast<TIdx>(maxBlockDim[1u]),
                                alpaka::core::clipCast<TIdx>(maxBlockDim[0u]))),
                        // m_blockThreadCountMax
                        alpaka::core::clipCast<TIdx>(maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()
                    };

#else
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_HIP_RT_CHECK(hipGetDeviceProperties(
                        &hipDevProp,
                        dev.m_iDevice));

                    return {
                        // m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(hipDevProp.multiProcessorCount),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxGridSize[2u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxGridSize[1u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsDim[2u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsDim[1u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsDim[0u]))),
                        // m_blockThreadCountMax
                        alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()
                    };
#endif
                }
            };
            //#############################################################################
            //! The GPU CUDA accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccGpuCudaHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccGpuCudaHipRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
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
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccGpuCudaHipRt<TDim, TIdx>>
            {
                using type = dev::DevCudaHipRt;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccGpuCudaHipRt<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace kernel
    {
        namespace detail
        {
            //#############################################################################
            //! specialization of the TKernelFnObj return type evaluation
            //
            // It is not possible to determine the result type of a __device__ lambda for CUDA on the host side.
            // https://github.com/ComputationalRadiationPhysics/alpaka/pull/695#issuecomment-446103194
            // The execution task TaskKernelGpuCudaHipRt is therefore performing this check on device side.
            template<
                typename TDim,
                typename TIdx>
            struct CheckFnReturnType<
                acc::AccGpuCudaHipRt<
                    TDim,
                    TIdx>>
            {
                template<
                    typename TKernelFnObj,
                    typename... TArgs>
                void operator()(
                    TKernelFnObj const &,
                    TArgs const & ...)
                {

                }
            };
        }

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
                acc::AccGpuCudaHipRt<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> kernel::TaskKernelGpuCudaHipRt<
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs...>
#endif
                {
                    return
                        kernel::TaskKernelGpuCudaHipRt<
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
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU CUDA execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccGpuCudaHipRt<TDim, TIdx>>
            {
                using type = pltf::PltfCudaHipRt;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccGpuCudaHipRt<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
