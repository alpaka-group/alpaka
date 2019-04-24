/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka
{
    //! The test specifics.
    namespace test
    {
        namespace traits
        {
            //! The default queue type trait for devices.
            template<typename TDev, typename TSfinae = void>
            struct DefaultQueueType;

            //! The default queue type trait specialization for the CPU device.
            template<>
            struct DefaultQueueType<alpaka::DevCpu>
            {
#if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::QueueCpuBlocking;
#else
                using type = alpaka::QueueCpuNonBlocking;
#endif
            };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

            //! The default queue type trait specialization for the CUDA/HIP device.
            template<>
            struct DefaultQueueType<alpaka::DevUniformCudaHipRt>
            {
#    if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::QueueUniformCudaHipRtBlocking;
#    else
                using type = alpaka::QueueUniformCudaHipRtNonBlocking;
#    endif
            };
#endif
        } // namespace traits
        //! The queue type that should be used for the given accelerator.
        template<typename TAcc>
        using DefaultQueue = typename traits::DefaultQueueType<TAcc>::type;

        namespace traits
        {
            //! The blocking queue trait.
            template<typename TQueue, typename TSfinae = void>
            struct IsBlockingQueue;

            //! The blocking queue trait specialization for a blocking CPU queue.
            template<typename TDev>
            struct IsBlockingQueue<alpaka::QueueGenericThreadsBlocking<TDev>>
            {
                static constexpr bool value = true;
            };

            //! The blocking queue trait specialization for a non-blocking CPU queue.
            template<typename TDev>
            struct IsBlockingQueue<alpaka::QueueGenericThreadsNonBlocking<TDev>>
            {
                static constexpr bool value = false;
            };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

            //! The blocking queue trait specialization for a blocking CUDA/HIP RT queue.
            template<>
            struct IsBlockingQueue<alpaka::QueueUniformCudaHipRtBlocking>
            {
                static constexpr bool value = true;
            };

            //! The blocking queue trait specialization for a non-blocking CUDA/HIP RT queue.
            template<>
            struct IsBlockingQueue<alpaka::QueueUniformCudaHipRtNonBlocking>
            {
                static constexpr bool value = false;
            };
#endif

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

            //! The default queue type trait specialization for the Omp5 device.
            template<>
            struct DefaultQueueType<alpaka::DevOmp5>
            {
#    if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::QueueOmp5Blocking;
#    else
                using type = alpaka::QueueOmp5Blocking;
#    endif
            };
#elif defined ALPAKA_ACC_ANY_BT_OACC_ENABLED

            //! The default queue type trait specialization for the OMP4 device.
            template<>
            struct DefaultQueueType<alpaka::DevOacc>
            {
                using type = alpaka::QueueOaccBlocking;
            };
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
#    ifdef ALPAKA_SYCL_BACKEND_ONEAPI
#        ifdef ALPAKA_SYCL_ONEAPI_CPU
            //! The default queue type trait specialization for the Intel CPU device.
            template<>
            struct DefaultQueueType<alpaka::experimental::DevCpuSyclIntel>
            {
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::experimental::QueueCpuSyclIntelBlocking;
#            else
                using type = alpaka::experimental::QueueCpuSyclIntelNonBlocking;
#            endif
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueCpuSyclIntelBlocking>
            {
                static constexpr auto value = true;
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueCpuSyclIntelNonBlocking>
            {
                static constexpr auto value = false;
            };
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_FPGA
            //! The default queue type trait specialization for the Intel SYCL device.
            template<>
            struct DefaultQueueType<alpaka::experimental::DevFpgaSyclIntel>
            {
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::experimental::QueueFpgaSyclIntelBlocking;
#            else
                using type = alpaka::experimental::QueueFpgaSyclIntelNonBlocking;
#            endif
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueFpgaSyclIntelBlocking>
            {
                static constexpr auto value = true;
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueFpgaSyclIntelNonBlocking>
            {
                static constexpr auto value = false;
            };
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_GPU
            //! The default queue type trait specialization for the Intel CPU device.
            template<>
            struct DefaultQueueType<alpaka::experimental::DevGpuSyclIntel>
            {
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::experimental::QueueGpuSyclIntelBlocking;
#            else
                using type = alpaka::experimental::QueueGpuSyclIntelNonBlocking;
#            endif
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueGpuSyclIntelBlocking>
            {
                static constexpr auto value = true;
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueGpuSyclIntelNonBlocking>
            {
                static constexpr auto value = false;
            };
#        endif
#    endif
#    ifdef ALPAKA_SYCL_BACKEND_XILINX
            //! The default queue type trait specialization for the Xilinx SYCL device.
            template<>
            struct DefaultQueueType<alpaka::experimental::DevFpgaSyclXilinx>
            {
#        if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::experimental::QueueFpgaSyclXilinxBlocking;
#        else
                using type = alpaka::experimental::QueueFpgaSyclXilinxNonBlocking;
#        endif
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueFpgaSyclXilinxBlocking>
            {
                static constexpr auto value = true;
            };

            template<>
            struct IsBlockingQueue<alpaka::experimental::QueueFpgaSyclXilinxNonBlocking>
            {
                static constexpr auto value = false;
            };
#    endif
#endif
        } // namespace traits
        //! The queue type that should be used for the given accelerator.
        template<typename TQueue>
        using IsBlockingQueue = traits::IsBlockingQueue<TQueue>;

        //! A std::tuple holding tuples of devices and corresponding queue types.
        using TestQueues = std::tuple<
            std::tuple<alpaka::DevCpu, alpaka::QueueCpuBlocking>,
            std::tuple<alpaka::DevCpu, alpaka::QueueCpuNonBlocking>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            ,
            std::tuple<alpaka::DevUniformCudaHipRt, alpaka::QueueUniformCudaHipRtBlocking>,
            std::tuple<alpaka::DevUniformCudaHipRt, alpaka::QueueUniformCudaHipRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
            ,
            std::tuple<alpaka::DevHipRt, alpaka::QueueHipRtBlocking>,
            std::tuple<alpaka::DevHipRt, alpaka::QueueHipRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
            ,
            std::tuple<alpaka::DevOmp5, alpaka::QueueOmp5Blocking>,
            std::tuple<alpaka::DevOmp5, alpaka::QueueOmp5NonBlocking>
#elif defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED)
            ,
            std::tuple<alpaka::DevOacc, alpaka::QueueOaccBlocking>,
            std::tuple<alpaka::DevOacc, alpaka::QueueOaccNonBlocking>
#endif
#ifdef ALPAKA_ACC_SYCL_ENABLED
#    ifdef ALPAKA_SYCL_BACKEND_ONEAPI
#        ifdef ALPAKA_SYCL_ONEAPI_CPU
            ,
            std::tuple<alpaka::experimental::DevCpuSyclIntel, alpaka::experimental::QueueCpuSyclIntelBlocking>,
            std::tuple<alpaka::experimental::DevCpuSyclIntel, alpaka::experimental::QueueCpuSyclIntelNonBlocking>
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_FPGA
            ,
            std::tuple<alpaka::experimental::DevFpgaSyclIntel, alpaka::experimental::QueueFpgaSyclIntelBlocking>,
            std::tuple<alpaka::experimental::DevFpgaSyclIntel, alpaka::experimental::QueueFpgaSyclIntelNonBlocking>
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_GPU
            ,
            std::tuple<alpaka::experimental::DevGpuSyclIntel, alpaka::experimental::QueueGpuSyclIntelBlocking>,
            std::tuple<alpaka::experimental::DevGpuSyclIntel, alpaka::experimental::QueueGpuSyclIntelNonBlocking>
#        endif
#    endif
#    if defined(ALPAKA_SYCL_BACKEND_XILINX)
            ,
            std::tuple<alpaka::experimental::DevFpgaSyclXilinx, alpaka::experimental::QueueFpgaSyclXilinxBlocking>,
            std::tuple<alpaka::experimental::DevFpgaSyclXilinx, alpaka::experimental::QueueFpgaSyclXilinxNonBlocking>
#    endif
#endif
            >;
    } // namespace test
} // namespace alpaka
