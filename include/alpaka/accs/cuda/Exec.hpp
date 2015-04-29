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
#include <alpaka/accs/cuda/Common.hpp>
#include <alpaka/accs/cuda/Acc.hpp>         // AccCuda
#include <alpaka/accs/cuda/Dev.hpp>         // DevCuda
#include <alpaka/accs/cuda/Event.hpp>       // EventCuda
#include <alpaka/accs/cuda/Stream.hpp>      // StreamCuda
#include <alpaka/traits/Kernel.hpp>         // BlockSharedExternMemSizeBytes

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
                //! The CUDA kernel entry point.
                // \NOTE: A __global__ function or function template cannot have a trailing return type.
                //-----------------------------------------------------------------------------
                template<
                    typename TKernelFunctor,
                    typename... TArgs>
                __global__ void cudaKernel(
                    TKernelFunctor kernelFunctor,
                    TArgs ... args)
                {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
        #error "Cuda device capability >= 2.0 is required!"
#endif
                    AccCuda acc;

                    kernelFunctor(
                        acc,
                        args...);
                }

                //#############################################################################
                //! The CUDA accelerator executor.
                //#############################################################################
                class ExecCuda
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecCuda(
                        TWorkDiv const & workDiv,
                        StreamCuda & stream) :
                            m_Stream(stream),
                            m_v3uiGridBlockExtents(workdiv::getWorkDiv<Grid, Blocks, dim::Dim3>(workDiv)),
                            m_v3uiBlockThreadExtents(workdiv::getWorkDiv<Block, Threads, dim::Dim3>(workDiv))
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCuda(ExecCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCuda(ExecCuda &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCuda const &) -> ExecCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecCuda() = default;
#else
                    ALPAKA_FCT_HOST virtual ~ExecCuda() noexcept = default;
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
                        // TODO: Check that (sizeof(TKernelFunctor) * m_v3uiBlockThreadExtents.prod()) < available memory size

                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //std::size_t uiPrintfFifoSize;
                        //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                        //std::cout << "uiPrintfFifoSize: " << uiPrintfFifoSize << std::endl;
                        //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, uiPrintfFifoSize*10);
                        //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                        //std::cout << "uiPrintfFifoSize: " <<  uiPrintfFifoSize << std::endl;
#endif

                        dim3 const gridDim(
                            static_cast<unsigned int>(m_v3uiGridBlockExtents[0u]),
                            static_cast<unsigned int>(m_v3uiGridBlockExtents[1u]),
                            static_cast<unsigned int>(m_v3uiGridBlockExtents[2u]));
                        dim3 const blockDim(
                            static_cast<unsigned int>(m_v3uiBlockThreadExtents[0u]),
                            static_cast<unsigned int>(m_v3uiBlockThreadExtents[1u]),
                            static_cast<unsigned int>(m_v3uiBlockThreadExtents[2u]));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << "v3uiBlockThreadExtents: " <<  gridDim.x << " " <<  gridDim.y << " " <<  gridDim.z << std::endl;
                        std::cout << "v3uiBlockThreadExtents: " <<  blockDim.x << " " <<  blockDim.y << " " <<  blockDim.z << std::endl;
#endif

                        // Get the size of the block shared extern memory.
                        auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccCuda>(
                            m_v3uiBlockThreadExtents,
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
                        cudaFuncGetAttributes(&funcAttrs, cudaKernel<TKernelFunctor, TArgs...>);
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
                        cudaKernel<TKernelFunctor, TArgs...><<<
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
                    Vec3<> const m_v3uiGridBlockExtents;
                    Vec3<> const m_v3uiBlockThreadExtents;

                public:
                    StreamCuda m_Stream;
                };
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CUDA accelerator executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::cuda::detail::ExecCuda>
            {
                using type = accs::cuda::detail::AccCuda;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CUDA accelerator executor event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::cuda::detail::ExecCuda>
            {
                using type = accs::cuda::detail::EventCuda;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CUDA accelerator executor executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::cuda::detail::ExecCuda>
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
                accs::cuda::detail::ExecCuda>
            {
                using type = accs::cuda::detail::DevCuda;
            };
            //#############################################################################
            //! The CUDA accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::cuda::detail::ExecCuda>
            {
                using type = accs::cuda::detail::DevManCuda;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CUDA accelerator executor stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::cuda::detail::ExecCuda>
            {
                using type = accs::cuda::detail::StreamCuda;
            };
            //#############################################################################
            //! The CUDA accelerator executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::cuda::detail::ExecCuda>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::cuda::detail::ExecCuda const & exec)
                -> accs::cuda::detail::StreamCuda
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
