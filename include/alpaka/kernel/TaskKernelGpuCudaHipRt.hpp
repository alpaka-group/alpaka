/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

// Implementation details.
#include <alpaka/acc/AccGpuCudaHipRt.hpp>
#include <alpaka/dev/DevCudaHipRt.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/queue/QueueCudaHipRtNonBlocking.hpp>
#include <alpaka/queue/QueueCudaHipRtBlocking.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <alpaka/acc/Traits.hpp>
    #include <alpaka/dev/Traits.hpp>
    #include <alpaka/workdiv/WorkDivHelpers.hpp>
#endif

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace kernel
    {
        namespace cuda
        {
            namespace detail
            {
                //-----------------------------------------------------------------------------
                //! The GPU CUDA-HIP kernel entry point.
                // \NOTE: 'A __global__ function or function template cannot have a trailing return type.'
                template<
                    typename TDim,
                    typename TIdx,
                    typename TKernelFnObj,
                    typename... TArgs>
                __global__ void cudaHipKernel(
                    vec::Vec<TDim, TIdx> const threadElemExtent,
                    TKernelFnObj const kernelFnObj,
                    TArgs ... args)
                {
#if BOOST_ARCH_PTX && (BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(2, 0, 0))
    #error "CudaHip device capability >= 2.0 is required!"
#endif

// with clang it is not possible to query std::result_of for a pure device lambda created on the host side
//TODO decide what to do with clang HIP
#if !(BOOST_COMP_CLANG_CUDA && BOOST_COMP_CLANG)
                    static_assert(
                        std::is_same<typename std::result_of<
                            TKernelFnObj(acc::AccGpuCudaHipRt<TDim, TIdx> const &, TArgs const & ...)>::type, void>::value,
                        "The TKernelFnObj is required to return void!");
#endif
                    acc::AccGpuCudaHipRt<TDim, TIdx> acc(threadElemExtent);

                    kernelFnObj(
                        const_cast<acc::AccGpuCudaHipRt<TDim, TIdx> const &>(acc),
                        args...);
                }

                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TIdx
                >
                ALPAKA_FN_HOST auto checkVecOnly3Dim(
                    vec::Vec<TDim, TIdx> const & vec)
                -> void
                {
                    for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i<TDim::value; ++i)
                    {
                        if(vec[TDim::value-1u-i] != 1)
                        {
                            throw std::runtime_error("The CUDA/HIP accelerator supports a maximum of 3 dimensions. All work division extents of the dimensions higher 3 have to be 1!");
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TIdx
                >
                ALPAKA_FN_HOST auto convertVecToCudaHipDim(
                    vec::Vec<TDim, TIdx> const & vec)
                -> dim3
                {
                    dim3 dim(1, 1, 1);
                    for(auto i(static_cast<typename TDim::value_type>(0)); i<std::min(static_cast<typename TDim::value_type>(3), TDim::value); ++i)
                    {
                        reinterpret_cast<unsigned int *>(&dim)[i] = static_cast<unsigned int>(vec[TDim::value-1u-i]);
                    }
                    checkVecOnly3Dim(vec);
                    return dim;
                }
            }
        }

        //#############################################################################
        //! The GPU-HIP CUDA accelerator execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelGpuCudaHipRt final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
// gcc-4.9 libstdc++ does not support std::is_trivially_copyable.
// MSVC std::is_trivially_copyable seems to be buggy (last tested at 15.7).
// libc++ in combination with CUDA does not seem to work.
#if (!BOOST_COMP_MSVC) && !(defined(__GLIBCXX__) && (__GLIBCXX__))
&& !(defined(_LIBCPP_VERSION) && (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA))
            static_assert(
                meta::Conjunction<
                    std::is_trivially_copyable<
                        TKernelFnObj>,
                    std::is_trivially_copyable<
                        TArgs>...
                    >::value,
                "The given kernel function object and its arguments have to fulfill is_trivially_copyable!");
#endif

            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelGpuCudaHipRt(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs && ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(std::forward<TArgs>(args)...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelGpuCudaHipRt(TaskKernelGpuCudaHipRt const &) = default;
            //-----------------------------------------------------------------------------
            TaskKernelGpuCudaHipRt(TaskKernelGpuCudaHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelGpuCudaHipRt const &) -> TaskKernelGpuCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelGpuCudaHipRt &&) -> TaskKernelGpuCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelGpuCudaHipRt() = default;

            TKernelFnObj m_kernelFnObj;
            std::tuple<typename std::decay<TArgs>::type...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA-HIP execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccGpuCudaHipRt<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA-HIP execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The GPU CUDA-HIP execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU CUDA-HIP execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The GPU CUDA-HIP execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP non-blocking kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueCudaHipRtNonBlocking,
                kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtNonBlocking & queue,
                    kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 const gridDim(kernel::cuda::detail::convertVecToCudaHipDim(gridBlockExtent));
                    dim3 const blockDim(kernel::cuda::detail::convertVecToCudaHipDim(blockThreadExtent));
                    kernel::cuda::detail::checkVecOnly3Dim(threadElemExtent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                        << " gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x
                        << " blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x
                        << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaHipRt<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaHipRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](typename std::decay<TArgs>::type const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaHipRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory idx.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, kernel::cuda::detail::cudaHipKernel<TDim, TIdx, TKernelFnObj, TArgs...>);
                    std::cout << __func__
                        << " binaryVersion: " << funcAttrs.binaryVersion
                        << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                        << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                        << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                        << " numRegs: " << funcAttrs.numRegs
                        << " ptxVersion: " << funcAttrs.ptxVersion
                        << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                        << std::endl;
    #endif
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
#else
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
#endif
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](typename std::decay<TArgs>::type const & ... args)
                        {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            kernel::cuda::detail::cudaHipKernel<TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type...><<<

                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_CudaHipQueue>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
#else
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(kernel::hip::detail::hipKernel< TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type... >),
                                gridDim,
                                blockDim,
                                static_cast<std::uint32_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_HipQueue,
                                hipLaunchParm{},
                                threadElemExtent,
                                task.m_kernelFnObj,
                                args...
                            );
#endif
                        },
                        task.m_args);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaHipQueue);
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::cuda::detail::cudaRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
    #else
                    hipStreamSynchronize(
                        queue.m_spQueueImpl->m_HipQueue);
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::hip::detail::hipRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
    #endif
#endif
                }
            };
            //#############################################################################
            //! The CUDA synchronous kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueCudaHipRtBlocking,
                kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtBlocking & queue,
                    kernel::TaskKernelGpuCudaHipRt<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    dim3 const gridDim(kernel::cuda::detail::convertVecToCudaHipDim(gridBlockExtent));
                    dim3 const blockDim(kernel::cuda::detail::convertVecToCudaHipDim(blockThreadExtent));
                    kernel::cuda::detail::checkVecOnly3Dim(threadElemExtent);
#else
                    dim3 gridDim(kernel::hip::detail::convertVecToHipDim(gridBlockExtent));
                    dim3 blockDim(kernel::hip::detail::convertVecToHipDim(blockThreadExtent));
                    kernel::hip::detail::checkVecOnly3Dim(threadElemExtent);
#endif


#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__ << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                    std::cout << __func__ << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaHipRt<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaHipRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](typename std::decay<TArgs>::type const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaHipRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory idx.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
    #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, kernel::cuda::detail::Hip<TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type...>);
    #else
                    //TODO to be decided (original is out-commented
    #endif
                    std::cout << __func__
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
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
#else
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
#endif
                    // Enqueue the kernel execution.
                    meta::apply(
                        [&](typename std::decay<TArgs>::type const & ... args)
                        {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            kernel::cuda::detail::cudaHipKernel<TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type...><<<
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_CudaHipQueue>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
#else
                            hipLaunchKernel(
                                HIP_KERNEL_NAME(kernel::hip::detail::hipKernel< TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type... >),
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_HipQueue,
                                threadElemExtent,
                                task.m_kernelFnObj,
                                args...
                            );
#endif
                        },
                        task.m_args);

                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    cudaStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaHipQueue);
#else
                    hipStreamSynchronize(
                        queue.m_spQueueImpl->m_HipQueue);
#endif
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
    #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ::alpaka::cuda::detail::cudaRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
    #else
                    ::alpaka::hip::detail::hipRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
    #endif
#endif
                }
            };
        }
    }
}

#endif
