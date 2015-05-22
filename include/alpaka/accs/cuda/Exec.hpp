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

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/accs/cuda/Acc.hpp>         // AccGpuCuda
#include <alpaka/devs/cuda/Dev.hpp>         // DevCuda
#include <alpaka/devs/cuda/Event.hpp>       // EventCuda
#include <alpaka/devs/cuda/Stream.hpp>      // StreamCuda
#include <alpaka/traits/Kernel.hpp>         // BlockSharedExternMemSizeBytes
#include <alpaka/core/Cuda.hpp>             // ALPAKA_CUDA_RT_CHECK

#include <boost/predef.h>                   // workarounds

#include <stdexcept>                        // std::runtime_error
#include <utility>                          // std::forward
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                     // std::cout
#endif

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                //-----------------------------------------------------------------------------
                //! The GPU CUDA kernel entry point.
                // \NOTE: A __global__ function or function template cannot have a trailing return type.
                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TKernelFunctor,
                    typename... TArgs>
                __global__ void cudaKernel(
                    TKernelFunctor kernelFunctor,
                    TArgs ... args)
                {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
        #error "Cuda device capability >= 2.0 is required!"
#endif
                    AccGpuCuda<TDim> acc;

                    kernelFunctor(
                        acc,
                        args...);
                }

                //#############################################################################
                //! The GPU CUDA accelerator executor.
                //#############################################################################
                template<
                    typename TDim>
                class ExecGpuCuda
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecGpuCuda(
                        TWorkDiv const & workDiv,
                        devs::cuda::StreamCuda & stream) :
                            m_Stream(stream),
                            m_vuiGridBlockExtents(workdiv::getWorkDiv<Grid, Blocks>(workDiv)),
                            m_vuiBlockThreadExtents(workdiv::getWorkDiv<Block, Threads>(workDiv))
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecGpuCuda(ExecGpuCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecGpuCuda(ExecGpuCuda &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecGpuCuda const &) -> ExecGpuCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecGpuCuda() = default;
#else
                    ALPAKA_FCT_HOST virtual ~ExecGpuCuda() noexcept = default;
#endif

                    //-----------------------------------------------------------------------------
                    //! Executes the kernel functor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto operator()(
                        // \NOTE: No universal reference (&&) or const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                        // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                        // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                        TKernelFunctor kernelFunctor,
                        TArgs ... args) const
                    -> void
                    {
#if (!__GLIBCXX__) // libstdc++ even for gcc-4.9 does not support std::is_trivially_copyable.
                        static_assert(std::is_trivially_copyable<TKernelFunctor>::value, "The given kernel functor has to fulfill is_trivially_copyable!");
#endif
                        // TODO: Check that (sizeof(TKernelFunctor) * m_3uiBlockThreadExtents.prod()) < available memory size

                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //std::size_t uiPrintfFifoSize;
                        //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                        //std::cout << "uiPrintfFifoSize: " << uiPrintfFifoSize << std::endl;
                        //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, uiPrintfFifoSize*10);
                        //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                        //std::cout << "uiPrintfFifoSize: " <<  uiPrintfFifoSize << std::endl;
#endif
                        dim3 gridDim(1u, 1u, 1u);
                        dim3 blockDim(1u, 1u, 1u);
                        for(std::size_t i(0u); i<3u; ++i)
                        {
                            reinterpret_cast<unsigned int *>(&gridDim)[i] = m_vuiGridBlockExtents[TDim::value-1u-i];
                            reinterpret_cast<unsigned int *>(&blockDim)[i] = m_vuiBlockThreadExtents[TDim::value-1u-i];
                        }
                        for(std::size_t i(3u); i<TDim::value; ++i)
                        {
                            assert(m_vuiGridBlockExtents[TDim::value-1u-i] == 1);
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                        std::cout << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

                        // Get the size of the block shared extern memory.
                        auto const uiBlockSharedExternMemSizeBytes(
                            kernel::getBlockSharedExternMemSizeBytes<
                                typename std::decay<TKernelFunctor>::type,
                                AccGpuCuda<TDim>>(
                                    m_vuiBlockThreadExtents,
                                    std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        // Log the block shared memory size.
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        // Log the function attributes.
                        cudaFuncAttributes funcAttrs;
                        cudaFuncGetAttributes(&funcAttrs, cudaKernel<TDim, TKernelFunctor, TArgs...>);
                        std::cout << BOOST_CURRENT_FUNCTION
                            << "binaryVersion: " << funcAttrs.binaryVersion
                            << "constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                            << "localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                            << "maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                            << "numRegs: " << funcAttrs.numRegs
                            << "ptxVersion: " << funcAttrs.ptxVersion
                            << "sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                            << std::endl;
#endif

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_Stream.m_Dev.m_iDevice));
                        // Enqueue the kernel execution.
                        cudaKernel<TDim, TKernelFunctor, TArgs...><<<
                            gridDim,
                            blockDim,
                            uiBlockSharedExternMemSizeBytes,
                            *m_Stream.m_spCudaStream.get()>>>(
                                kernelFunctor,
                                args...);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        // Wait for the kernel execution to finish but do not check error return of this call.
                        // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                        cudaStreamSynchronize(*m_Stream.m_spCudaStream.get());
                        //cudaDeviceSynchronize();
                        cudaError_t const error(cudaGetLastError());
                        if(error != cudaSuccess)
                        {
                            std::string const sError("The execution of kernel '" + std::string(typeid(TKernelFunctor).name()) + " failed with error: '" + std::string(cudaGetErrorString(error)) + "'");
                            std::cerr << sError << std::endl;
                            ALPAKA_DEBUG_BREAK;
                            throw std::runtime_error(sError);
                        }
#endif
                    }

                private:
                    Vec<TDim> const m_vuiGridBlockExtents;
                    Vec<TDim> const m_vuiBlockThreadExtents;

                public:
                    devs::cuda::StreamCuda m_Stream;
                };
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The GPU CUDA executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                using type = accs::cuda::detail::AccGpuCuda<TDim>;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The GPU CUDA executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                using type = devs::cuda::DevCuda;
            };
            //#############################################################################
            //! The GPU CUDA executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                using type = devs::cuda::DevManCuda;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The GPU CUDA executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                using type = TDim;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The GPU CUDA executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct EventType<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                using type = devs::cuda::EventCuda;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The GPU CUDA executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                using type = accs::cuda::detail::ExecGpuCuda<TDim>;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The GPU CUDA executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct StreamType<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                using type = devs::cuda::StreamCuda;
            };
            //#############################################################################
            //! The GPU CUDA executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetStream<
                accs::cuda::detail::ExecGpuCuda<TDim>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::cuda::detail::ExecGpuCuda<TDim> const & exec)
                -> devs::cuda::StreamCuda
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
