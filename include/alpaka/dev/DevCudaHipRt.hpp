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
        class PltfCudaHipRt;
    }

    namespace dev
    {
        //#############################################################################
        //! The CUDA-HIP RT device handle.
        class DevCudaHipRt
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfCudaHipRt>;

        protected:
            //-----------------------------------------------------------------------------
            DevCudaHipRt() = default;
        public:
            //-----------------------------------------------------------------------------
            DevCudaHipRt(DevCudaHipRt const &) = default;
            //-----------------------------------------------------------------------------
            DevCudaHipRt(DevCudaHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCudaHipRt const &) -> DevCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCudaHipRt &&) -> DevCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevCudaHipRt const & rhs) const
            -> bool
            {
                return m_iDevice == rhs.m_iDevice;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevCudaHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevCudaHipRt() = default;

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
                dev::DevCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevCudaHipRt const & dev)
                -> std::string
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
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
                dev::DevCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevCudaHipRt const & dev)
                -> std::size_t
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
#else
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
#endif

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemGetInfo(
                            &freeInternal,
                            &totalInternal));
#else
                    // \TODO: Check which is faster: hipMemGetInfo().totalInternal vs hipGetDeviceProperties().totalGlobalMem
                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));
#endif

                    return totalInternal;
                }
            };

            //#############################################################################
            //! The CUDA-HIP RT device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevCudaHipRt const & dev)
                -> std::size_t
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
#else
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
#endif
                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemGetInfo(
                            &freeInternal,
                            &totalInternal));
#else
                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));
#endif

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The CUDA-HIP RT device reset trait specialization.
            template<>
            struct Reset<
                dev::DevCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevCudaHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(
                        cudaDeviceReset());
#else
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
            class BufCudaHipRt;

            namespace traits
            {
                //#############################################################################
                //! The CUDA-HIP RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevCudaHipRt,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufCudaHipRt<TElem, TDim, TIdx>;
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
                dev::DevCudaHipRt>
            {
                using type = pltf::PltfCudaHipRt;
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
                dev::DevCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevCudaHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceSynchronize());
#else
                    ALPAKA_HIP_RT_CHECK(hipSetDevice(
                        dev.m_iDevice));
                    ALPAKA_HIP_RT_CHECK(hipDeviceSynchronize());
#endif
                }
            };
        }
    }
}

#endif
