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
#include <alpaka/acc/Traits.hpp>            // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>            // dev::traits::DevType
#include <alpaka/dim/Traits.hpp>            // dim::traits::DimType
#include <alpaka/event/Traits.hpp>          // event::traits::EventType
#include <alpaka/exec/Traits.hpp>           // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>           // size::traits::SizeType
#include <alpaka/stream/Traits.hpp>         // stream::traits::StreamType

// Implementation details.
#include <alpaka/acc/AccGpuCudaRt.hpp>      // acc:AccGpuCudaRt
#include <alpaka/dev/DevCudaRt.hpp>         // dev::DevCudaRt
#include <alpaka/event/EventCudaRt.hpp>     // event::EventCudaRt
#include <alpaka/kernel/Traits.hpp>         // kernel::getBlockSharedExternMemSizeBytes
#include <alpaka/stream/StreamCudaRt.hpp>   // stream::StreamCudaRt
#include <alpaka/workdiv/WorkDivMembers.hpp>// workdiv::WorkDivMembers

#include <alpaka/core/Cuda.hpp>             // ALPAKA_CUDA_RT_CHECK

#include <boost/predef.h>                   // workarounds
#include <boost/mpl/apply.hpp>              // boost::mpl::apply
#include <boost/mpl/and.hpp>                // boost::mpl::and_

#include <stdexcept>                        // std::runtime_error
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                     // std::cout
#endif

namespace alpaka
{
    namespace exec
    {
        namespace cuda
        {
            namespace detail
            {
                //-----------------------------------------------------------------------------
                //! The GPU CUDA kernel entry point.
                // \NOTE: 'A __global__ function or function template cannot have a trailing return type.'
                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TSize,
                    typename TKernelFnObj,
                    typename... TArgs>
                __global__ void cudaKernel(
                    TKernelFnObj kernelFnObj,
                    TArgs ... args)
                {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error "Cuda device capability >= 2.0 is required!"
#endif
                    acc::AccGpuCudaRt<TDim, TSize> acc;

                    kernelFnObj(
                        const_cast<acc::AccGpuCudaRt<TDim, TSize> const &>(acc),
                        args...);
                }
            }
        }

        //#############################################################################
        //! The GPU CUDA accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class ExecGpuCudaRt final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecGpuCudaRt(
                TWorkDiv const & workDiv,
                stream::StreamCudaRt & stream) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    m_Stream(stream)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecGpuCudaRt(ExecGpuCudaRt const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecGpuCudaRt(ExecGpuCudaRt &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecGpuCudaRt const &) -> ExecGpuCudaRt & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecGpuCudaRt &&) -> ExecGpuCudaRt & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ExecGpuCudaRt() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            //-----------------------------------------------------------------------------
            template<
                typename TKernelFnObj,
                typename... TArgs>
            ALPAKA_FN_HOST auto operator()(
                // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                TKernelFnObj kernelFnObj,
                TArgs ... args) const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if (!__GLIBCXX__) // libstdc++ even for gcc-4.9 does not support std::is_trivially_copyable.
                static_assert(
                    boost::mpl::and_<
                        // This true_ is required for the zero argument case because and_ requires at least two arguments.
                        boost::mpl::true_,
                        std::is_trivially_copyable<
                            TKernelFnObj>,
                        boost::mpl::apply<
                            std::is_trivially_copyable<
                                boost::mpl::_1>,
                                TArgs>...
                        >::value,
                    "The given kernel function object and its arguments have to fulfill is_trivially_copyable!");
#endif
                // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtents.prod()) < available memory size

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                //std::size_t uiPrintfFifoSize;
                //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                //std::cout << BOOST_CURRENT_FUNCTION << "INFO: uiPrintfFifoSize: " << uiPrintfFifoSize << std::endl;
                //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, uiPrintfFifoSize*10);
                //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                //std::cout << BOOST_CURRENT_FUNCTION << "INFO: uiPrintfFifoSize: " <<  uiPrintfFifoSize << std::endl;
#endif

                auto const vuiGridBlockExtents(
                    workdiv::getWorkDiv<Grid, Blocks>(
                        *static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this)));
                auto const vuiBlockThreadExtents(
                    workdiv::getWorkDiv<Block, Threads>(
                        *static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this)));

                dim3 gridDim(1u, 1u, 1u);
                dim3 blockDim(1u, 1u, 1u);
                // \FIXME: CUDA currently supports a maximum of 3 dimensions!
                for(auto i(static_cast<typename TDim::value_type>(0)); i<std::min(static_cast<typename TDim::value_type>(3), TDim::value); ++i)
                {
                    reinterpret_cast<unsigned int *>(&gridDim)[i] = vuiGridBlockExtents[TDim::value-1u-i];
                    reinterpret_cast<unsigned int *>(&blockDim)[i] = vuiBlockThreadExtents[TDim::value-1u-i];
                }
                // Assert that all extents of the higher dimensions are 1!
                for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i<TDim::value; ++i)
                {
                    assert(vuiGridBlockExtents[TDim::value-1u-i] == 1);
                    assert(vuiBlockThreadExtents[TDim::value-1u-i] == 1);
                }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                std::cout << BOOST_CURRENT_FUNCTION << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

                // Get the size of the block shared extern memory.
                auto const uiBlockSharedExternMemSizeBytes(
                    kernel::getBlockSharedExternMemSizeBytes<
                        typename std::decay<TKernelFnObj>::type,
                        acc::AccGpuCudaRt<TDim, TSize>>(
                            vuiBlockThreadExtents,
                            args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the block shared memory size.
                std::cout << BOOST_CURRENT_FUNCTION
                    << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                    << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the function attributes.
                cudaFuncAttributes funcAttrs;
                cudaFuncGetAttributes(&funcAttrs, cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...>);
                std::cout << BOOST_CURRENT_FUNCTION
                    << " binaryVersion: " << funcAttrs.binaryVersion
                    << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                    << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                    << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                    << " numRegs: " << funcAttrs.numRegs
                    << " ptxVersion: " << funcAttrs.ptxVersion
                    << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                    << std::endl;
#endif

                // Set the current device.
                ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                    m_Stream.m_spStreamCudaImpl->m_Dev.m_iDevice));
                // Enqueue the kernel execution.
                cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...><<<
                    gridDim,
                    blockDim,
                    uiBlockSharedExternMemSizeBytes,
                    m_Stream.m_spStreamCudaImpl->m_CudaStream>>>(
                        kernelFnObj,
                        args...);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                // Wait for the kernel execution to finish but do not check error return of this call.
                // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                cudaStreamSynchronize(m_Stream.m_spStreamCudaImpl->m_CudaStream);
                //cudaDeviceSynchronize();
                cudaError_t const error(cudaGetLastError());
                if(error != cudaSuccess)
                {
                    std::string const sError("The execution of kernel '" + std::string(typeid(TKernelFnObj).name()) + "' failed with error: '" + std::string(cudaGetErrorString(error)) + "'");
                    std::cerr << sError << std::endl;
                    ALPAKA_DEBUG_BREAK;
                    throw std::runtime_error(sError);
                }
#endif
            }

        public:
            stream::StreamCudaRt m_Stream;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                using type = acc::AccGpuCudaRt<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The GPU CUDA executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                exec::ExecGpuCudaRt<TDim, TSize>>
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
            //! The GPU CUDA executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct EventType<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                using type = event::EventCudaRt;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                using type = exec::ExecGpuCudaRt<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct StreamType<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                using type = stream::StreamCudaRt;
            };
            //#############################################################################
            //! The GPU CUDA executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetStream<
                exec::ExecGpuCudaRt<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getStream(
                    exec::ExecGpuCudaRt<TDim, TSize> const & exec)
                -> stream::StreamCudaRt
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
