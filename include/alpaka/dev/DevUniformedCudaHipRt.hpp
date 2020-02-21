/* Copyright 2019 Benjamin Worpitz
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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/Properties.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

namespace alpaka
{
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfUniformedCudaHipRt;
    }
    namespace queue
    {
        class QueueUniformedCudaHipRtBlocking;
        class QueueUniformedCudaHipRtNonBlocking;
    }

    namespace dev
    {
        //#############################################################################
        //! The CUDA-HIP RT device handle.
        class DevUniformedCudaHipRt : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevUniformedCudaHipRt>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfUniformedCudaHipRt>;

        protected:
            //-----------------------------------------------------------------------------
            DevUniformedCudaHipRt() = default;
        public:
            //-----------------------------------------------------------------------------
            DevUniformedCudaHipRt(DevUniformedCudaHipRt const &) = default;
            //-----------------------------------------------------------------------------
            DevUniformedCudaHipRt(DevUniformedCudaHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevUniformedCudaHipRt const &) -> DevUniformedCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevUniformedCudaHipRt &&) -> DevUniformedCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevUniformedCudaHipRt const & rhs) const
            -> bool
            {
                return m_iDevice == rhs.m_iDevice;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevUniformedCudaHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevUniformedCudaHipRt() = default;

        public:
            int m_iDevice;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT device name get trait specialization.
            template<>
            struct GetName<
                dev::DevUniformedCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevUniformedCudaHipRt const & dev)
                -> std::string
                {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    // There is cudaDeviceGetAttribute as faster alternative to cudaGetDeviceProperties to get a single device property but it has no option to get the name
                    cudaDeviceProp cudaDevProp;
                    ALPAKA_CUDA_RT_CHECK(
                        cudaGetDeviceProperties(
                            &cudaDevProp,
                            dev.m_iDevice));

                    return std::string(cudaDevProp.name);
#else
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_HIP_RT_CHECK(
                        hipGetDeviceProperties(
                            &hipDevProp,
                            dev.m_iDevice));

                    return std::string(hipDevProp.name);
#endif
                }
            };

            //#############################################################################
            //! The CUDA-HIP RT device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevUniformedCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevUniformedCudaHipRt const & dev)
                -> std::size_t
                {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return totalInternal;
#else
                  // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    // \TODO: Check which is faster: hipMemGetInfo().totalInternal vs hipGetDeviceProperties().totalGlobalMem
                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return totalInternal;
#endif
                }
            };

            //#############################################################################
            //! The CUDA-HIP RT device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevUniformedCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevUniformedCudaHipRt const & dev)
                -> std::size_t
                {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return freeInternal;
#else
                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return freeInternal;
#endif
                }
            };

            //#############################################################################
            //! The CUDA-HIP RT device reset trait specialization.
            template<>
            struct Reset<
                dev::DevUniformedCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevUniformedCudaHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(
                        cudaDeviceReset());
#else
                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
                    ALPAKA_HIP_RT_CHECK(
                        hipDeviceReset());
#endif
                }
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufUniformedCudaHipRt;

            namespace traits
            {
                //#############################################################################
                //! The CUDA-HIP RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevUniformedCudaHipRt,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufUniformedCudaHipRt<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevUniformedCudaHipRt>
            {
                using type = pltf::PltfUniformedCudaHipRt;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread CUDA-HIP device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevUniformedCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevUniformedCudaHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceSynchronize());
#else
                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(hipSetDevice(
                        dev.m_iDevice));
                    ALPAKA_HIP_RT_CHECK(hipDeviceSynchronize());
#endif
                }
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            template<>
            struct QueueType<
                dev::DevUniformedCudaHipRt,
                queue::Blocking
            >
            {
                using type = queue::QueueUniformedCudaHipRtBlocking;
            };

            template<>
            struct QueueType<
                dev::DevUniformedCudaHipRt,
                queue::NonBlocking
            >
            {
                using type = queue::QueueUniformedCudaHipRtNonBlocking;
            };
        }
    }
}

#endif
