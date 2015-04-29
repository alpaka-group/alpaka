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
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/accs/cuda/Common.hpp>
#include <alpaka/accs/cuda/Dev.hpp>         // DevCuda
#include <alpaka/accs/cuda/Event.hpp>       // EventCpu
#include <alpaka/accs/cuda/Stream.hpp>      // StreamCuda

#include <boost/predef.h>                   // workarounds

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator.
        //-----------------------------------------------------------------------------
        namespace cuda
        {
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator implementation details.
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

                class ExecCuda;

                //#############################################################################
                //! The CUDA accelerator.
                //!
                //! This accelerator allows parallel kernel execution on devices supporting CUDA.
                //#############################################################################
                class AccCuda :
                    protected WorkDivCuda,
                    private IdxCuda,
                    protected AtomicCuda
                {
                public:
                    /*template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    friend void ::alpaka::cuda::detail::cudaKernel(
                        TKernelFunctor kernelFunctor,
                        TArgs ... args);*/

                    //friend class ::alpaka::cuda::detail::ExecCuda;

                //private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AccCuda() :
                        WorkDivCuda(),
                        IdxCuda(),
                        AtomicCuda()
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda &&) = delete;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(AccCuda const &) -> AccCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~AccCuda() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getWorkDiv() const
                    -> Vec<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                            *static_cast<WorkDivCuda const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested indices.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getIdx() const
                    -> Vec<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit, TDim>(
                            *static_cast<IdxCuda const *>(this),
                            *static_cast<WorkDivCuda const *>(this));
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

    using AccCuda = accs::cuda::detail::AccCuda;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CUDA accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::AccCuda;
            };
            //#############################################################################
            //! The CUDA accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetAccDevProps<
                accs::cuda::detail::AccCuda>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    accs::cuda::detail::DevCuda const & dev)
                -> alpaka::acc::AccDevProps
                {
                    cudaDeviceProp cudaDevProp;
                    ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(
                        &cudaDevProp,
                        dev.m_iDevice));

                    return alpaka::acc::AccDevProps(
                        // m_uiMultiProcessorCount
                        static_cast<UInt>(cudaDevProp.multiProcessorCount),
                        // m_uiBlockThreadsCountMax
                        static_cast<UInt>(cudaDevProp.maxThreadsPerBlock),
                        // m_v3uiBlockThreadExtentsMax
                        Vec3<>(
                            static_cast<UInt>(cudaDevProp.maxThreadsDim[0]),
                            static_cast<UInt>(cudaDevProp.maxThreadsDim[1]),
                            static_cast<UInt>(cudaDevProp.maxThreadsDim[2])),
                        // m_v3uiGridBlockExtentsMax
                        Vec3<>(
                            static_cast<UInt>(cudaDevProp.maxGridSize[0]),
                            static_cast<UInt>(cudaDevProp.maxGridSize[1]),
                            static_cast<UInt>(cudaDevProp.maxGridSize[2])));
                        //devProps.m_uiMaxClockFrequencyHz = cudaDevProp.clockRate * 1000;
                }
            };
            //#############################################################################
            //! The CUDA accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::cuda::detail::AccCuda>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCuda";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CUDA accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::EventCuda;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CUDA accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::ExecCuda;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CUDA accelerator executor device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::DevCuda;
            };
            //#############################################################################
            //! The CUDA accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::DevManCuda;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CUDA accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::StreamCuda;
            };
        }
    }
}
