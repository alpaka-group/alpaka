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

// Base classes.
#include <alpaka/accs/cuda/WorkDiv.hpp>     // WorkDivCuda
#include <alpaka/accs/cuda/Idx.hpp>         // IdxCuda
#include <alpaka/accs/cuda/Atomic.hpp>      // AtomicCuda

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Dev.hpp>            // DevType

// Implementation details.
#include <alpaka/devs/cuda/Dev.hpp>         // DevCuda
#include <alpaka/core/Cuda.hpp>             // ALPAKA_CUDA_RT_CHECK

#include <boost/predef.h>                   // workarounds

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The GPU CUDA accelerator.
        //-----------------------------------------------------------------------------
        namespace cuda
        {
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                // Forward declarations.
                /*template<
                    typename TKernelFunctor,
                    typename... TArgs>
                __global__ void cudaKernel(
                    TKernelFunctor kernelFunctor,
                    TArgs ... args);*/

                template<
                    typename TDim>
                class ExecGpuCuda;

                //#############################################################################
                //! The GPU CUDA accelerator.
                //!
                //! This accelerator allows parallel kernel execution on devices supporting CUDA.
                //#############################################################################
                template<
                    typename TDim>
                class AccGpuCuda final :
                    protected WorkDivCuda<TDim>,
                    private IdxCuda<TDim>,
                    protected AtomicCuda
                {
                public:
                    /*template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    friend void ::alpaka::cuda::detail::cudaKernel(
                        TKernelFunctor kernelFunctor,
                        TArgs ... args);*/

                    //friend class ::alpaka::cuda::detail::ExecGpuCuda<TDim>;

                //private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AccGpuCuda() :
                        WorkDivCuda<TDim>(),
                        IdxCuda<TDim>(),
                        AtomicCuda()
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AccGpuCuda(AccGpuCuda const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AccGpuCuda(AccGpuCuda &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(AccGpuCuda const &) -> AccGpuCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(AccGpuCuda &&) -> AccGpuCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY ~AccGpuCuda() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The requested indices.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit>
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getIdx() const
                    -> Vec<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit>(
                            *static_cast<IdxCuda<TDim> const *>(this),
                            *static_cast<WorkDivCuda<TDim> const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit>
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getWorkDiv() const
                    -> Vec<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit>(
                            *static_cast<WorkDivCuda<TDim> const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Execute the atomic operation on the given address with the given value.
                    //! \return The old value before executing the atomic operation.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOp,
                        typename T>
                    ALPAKA_FCT_ACC auto atomicOp(
                        T * const addr,
                        T const & value) const
                    -> T
                    {
                        return atomic::atomicOp<TOp, T>(
                            addr,
                            value,
                            *static_cast<AtomicCuda const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto syncBlockThreads() const
                    -> void
                    {
                        __syncthreads();
                    }

                    //-----------------------------------------------------------------------------
                    //! \return Allocates block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T,
                        UInt TuiNumElements>
                    ALPAKA_FCT_ACC_CUDA_ONLY auto allocBlockSharedMem() const
                    -> T *
                    {
                        static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                        __shared__ T shMem[TuiNumElements];
                        return shMem;
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The pointer to the externally allocated block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T>
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getBlockSharedExternMem() const
                    -> T *
                    {
                        // Because unaligned access to variables is not allowed in device code,
                        // we have to use the widest possible type to have all types aligned correctly.
                        // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
                        // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types
                        extern __shared__ float4 shMem[];
                        return reinterpret_cast<T *>(shMem);
                    }
                };
            }
        }
    }

    template<
        typename TDim>
    using AccGpuCuda = accs::cuda::detail::AccGpuCuda<TDim>;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The GPU CUDA accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::cuda::detail::AccGpuCuda<TDim>>
            {
                using type = accs::cuda::detail::AccGpuCuda<TDim>;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                accs::cuda::detail::AccGpuCuda<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cuda::DevCuda const & dev)
                -> alpaka::acc::AccDevProps<TDim>
                {
                    cudaDeviceProp cudaDevProp;
                    ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(
                        &cudaDevProp,
                        dev.m_iDevice));

                    return {
                        // m_uiMultiProcessorCount
                        static_cast<UInt>(cudaDevProp.multiProcessorCount),
                        // m_uiBlockThreadsCountMax
                        static_cast<UInt>(cudaDevProp.maxThreadsPerBlock),
                        // m_vuiBlockThreadExtentsMax
                        alpaka::extent::getExtentsVecNd<TDim, UInt>(cudaDevProp.maxThreadsDim),
                        // m_vuiGridBlockExtentsMax
                        alpaka::extent::getExtentsVecNd<TDim, UInt>(cudaDevProp.maxGridSize)};
                }
            };
            //#############################################################################
            //! The GPU CUDA accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccName<
                accs::cuda::detail::AccGpuCuda<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccGpuCuda<" + std::to_string(TDim::value) + ">";
                }
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The GPU CUDA accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::cuda::detail::AccGpuCuda<TDim>>
            {
                using type = devs::cuda::DevCuda;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::cuda::detail::AccGpuCuda<TDim>>
            {
                using type = devs::cuda::DevManCuda;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The GPU CUDA accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::cuda::detail::AccGpuCuda<TDim>>
            {
                using type = TDim;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The GPU CUDA accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::cuda::detail::AccGpuCuda<TDim>>
            {
                using type = accs::cuda::detail::ExecGpuCuda<TDim>;
            };
        }
    }
}
