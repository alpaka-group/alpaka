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
#include <alpaka/acc/Traits.hpp>                // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                // dev::traits::DevType
#include <alpaka/dim/Traits.hpp>                // dim::traits::DimType
#include <alpaka/exec/Traits.hpp>               // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>               // size::traits::SizeType
#include <alpaka/stream/Traits.hpp>             // stream::traits::Enqueue

// Implementation details.
#include <alpaka/acc/AccGpuCudaRt.hpp>          // acc:AccGpuCudaRt
#include <alpaka/dev/DevCudaRt.hpp>             // dev::DevCudaRt
#include <alpaka/kernel/Traits.hpp>             // kernel::getBlockSharedExternMemSizeBytes
#include <alpaka/stream/StreamCudaRtAsync.hpp>  // stream::StreamCudaRtAsync
#include <alpaka/stream/StreamCudaRtSync.hpp>   // stream::StreamCudaRtSync
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <alpaka/acc/Traits.hpp>            // acc::getAccName
    #include <alpaka/dev/Traits.hpp>            // dev::getDev
    #include <alpaka/workdiv/WorkDivHelpers.hpp>// workdiv::isValidWorkDiv
#endif

#include <alpaka/core/Cuda.hpp>                 // ALPAKA_CUDA_RT_CHECK
#include <alpaka/core/ApplyTuple.hpp>           // core::Apply

#include <boost/predef.h>                       // workarounds
#include <boost/mpl/apply.hpp>                  // boost::mpl::apply
#include <boost/mpl/and.hpp>                    // boost::mpl::and_

#include <stdexcept>                            // std::runtime_error
#include <tuple>                                // std::tuple
#include <type_traits>                          // std::decay
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
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
                    TKernelFnObj const kernelFnObj,
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
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecGpuCudaRt final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
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

            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecGpuCudaRt(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TSize>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
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

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The GPU CUDA executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevManType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The GPU CUDA executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>,
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
            //! The GPU CUDA executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CUDA async device stream 1D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtents.prod()) < available memory size

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtents(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtents(
                        workdiv::getWorkDiv<Block, Threads>(task));

                    dim3 gridDim(1u, 1u, 1u);
                    dim3 blockDim(1u, 1u, 1u);
                    // \FIXME: CUDA currently supports a maximum of 3 dimensions!
                    for(auto i(static_cast<typename TDim::value_type>(0)); i<std::min(static_cast<typename TDim::value_type>(3), TDim::value); ++i)
                    {
                        reinterpret_cast<unsigned int *>(&gridDim)[i] = gridBlockExtents[TDim::value-1u-i];
                        reinterpret_cast<unsigned int *>(&blockDim)[i] = blockThreadExtents[TDim::value-1u-i];
                    }
                    // Assert that all extents of the higher dimensions are 1!
                    for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i<TDim::value; ++i)
                    {
                        assert(gridBlockExtents[TDim::value-1u-i] == 1);
                        assert(blockThreadExtents[TDim::value-1u-i] == 1);
                    }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                    std::cout << BOOST_CURRENT_FUNCTION << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaRt<TDim, TSize>>(dev::getDev(stream), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TSize>>() + "!");
                    }
#endif

                    // Get the size of the block shared extern memory.
                    auto const blockSharedExternMemSizeBytes(
                        core::apply(
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedExternMemSizeBytes<
                                        typename std::decay<TKernelFnObj>::type,
                                        acc::AccGpuCudaRt<TDim, TSize>>(
                                            blockThreadExtents,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory size.
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " BlockSharedExternMemSizeBytes: " << blockSharedExternMemSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...>);
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
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            stream.m_spStreamCudaRtAsyncImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    core::apply(
                        [&](TArgs ... args)
                        {
                            exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...><<<
                                gridDim,
                                blockDim,
                                blockSharedExternMemSizeBytes,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream>>>(
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        stream.m_spStreamCudaRtAsyncImpl->m_CudaStream);
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
            };
            //#############################################################################
            //! The CUDA sync device stream 1D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                stream::StreamCudaRtSync,
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtents.prod()) < available memory size

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtents(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtents(
                        workdiv::getWorkDiv<Block, Threads>(task));

                    dim3 gridDim(1u, 1u, 1u);
                    dim3 blockDim(1u, 1u, 1u);
                    // \FIXME: CUDA currently supports a maximum of 3 dimensions!
                    for(auto i(static_cast<typename TDim::value_type>(0)); i<std::min(static_cast<typename TDim::value_type>(3), TDim::value); ++i)
                    {
                        reinterpret_cast<unsigned int *>(&gridDim)[i] = gridBlockExtents[TDim::value-1u-i];
                        reinterpret_cast<unsigned int *>(&blockDim)[i] = blockThreadExtents[TDim::value-1u-i];
                    }
                    // Assert that all extents of the higher dimensions are 1!
                    for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i<TDim::value; ++i)
                    {
                        assert(gridBlockExtents[TDim::value-1u-i] == 1);
                        assert(blockThreadExtents[TDim::value-1u-i] == 1);
                    }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                    std::cout << BOOST_CURRENT_FUNCTION << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaRt<TDim, TSize>>(dev::getDev(stream), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TSize>>() + "!");
                    }
#endif

                    // Get the size of the block shared extern memory.
                    auto const blockSharedExternMemSizeBytes(
                        core::apply(
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedExternMemSizeBytes<
                                        TKernelFnObj,
                                        acc::AccGpuCudaRt<TDim, TSize>>(
                                            blockThreadExtents,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory size.
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " BlockSharedExternMemSizeBytes: " << blockSharedExternMemSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...>);
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
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            stream.m_spStreamCudaRtSyncImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    core::apply(
                        [&](TArgs ... args)
                        {
                            exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...><<<
                                gridDim,
                                blockDim,
                                blockSharedExternMemSizeBytes,
                                stream.m_spStreamCudaRtSyncImpl->m_CudaStream>>>(
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        stream.m_spStreamCudaRtSyncImpl->m_CudaStream);
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
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
            };
        }
    }
}
