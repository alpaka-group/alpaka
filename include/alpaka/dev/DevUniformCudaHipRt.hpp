/* Copyright 2022 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#if !defined(ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE)
#    error This is an internal header file, and should never be included directly.
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#error This file should not be included with ALPAKA_ACC_GPU_CUDA_ENABLED and ALPAKA_ACC_GPU_HIP_ENABLED both defined.
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(alpaka_dev_DevUniformCudaHipRt_hpp_CUDA)                         \
    || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(alpaka_dev_DevUniformCudaHipRt_hpp_HIP)

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(alpaka_dev_DevUniformCudaHipRt_hpp_CUDA)
#        define alpaka_dev_DevUniformCudaHipRt_hpp_CUDA
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(alpaka_dev_DevUniformCudaHipRt_hpp_HIP)
#        define alpaka_dev_DevUniformCudaHipRt_hpp_HIP
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/queue/Properties.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/traits/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    endif
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <cstddef>
#    include <string>
#    include <vector>

namespace alpaka
{
    namespace trait
    {
        template<typename TPltf, typename TSfinae>
        struct GetDevByIdx;
    }

    namespace ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE
    {
        template<typename TElem, typename TDim, typename TIdx>
        class BufUniformCudaHipRt;

        class PltfUniformCudaHipRt;
        class QueueUniformCudaHipRtBlocking;
        class QueueUniformCudaHipRtNonBlocking;

        //! The CUDA/HIP RT device handle.
        class DevUniformCudaHipRt
            : public concepts::Implements<ConceptCurrentThreadWaitFor, DevUniformCudaHipRt>
            , public concepts::Implements<ConceptDev, DevUniformCudaHipRt>
        {
            friend struct ::alpaka::trait::GetDevByIdx<PltfUniformCudaHipRt>;

        protected:
            DevUniformCudaHipRt() = default;

        public:
            ALPAKA_FN_HOST auto operator==(DevUniformCudaHipRt const& rhs) const -> bool
            {
                return m_iDevice == rhs.m_iDevice;
            }
            ALPAKA_FN_HOST auto operator!=(DevUniformCudaHipRt const& rhs) const -> bool
            {
                return !((*this) == rhs);
            }

            [[nodiscard]] auto getNativeHandle() const noexcept -> int
            {
                return m_iDevice;
            }

        private:
            DevUniformCudaHipRt(int iDevice) : m_iDevice(iDevice)
            {
            }
            int m_iDevice;
        };
    } // namespace ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE

    namespace trait
    {
        //! The CUDA/HIP RT device name get trait specialization.
        template<>
        struct GetName<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getName(ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt const& dev)
                -> std::string
            {
                // There is cuda/hip-DeviceGetAttribute as faster alternative to cuda/hip-GetDeviceProperties to get a
                // single device property but it has no option to get the name
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                cudaDeviceProp devProp;
#    endif
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                hipDeviceProp_t devProp;
#    endif
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(GetDeviceProperties)(&devProp, dev.getNativeHandle()));

                return std::string(devProp.name);
            }
        };

        //! The CUDA/HIP RT device available memory get trait specialization.
        template<>
        struct GetMemBytes<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getMemBytes(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt const& dev) -> std::size_t
            {
                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.getNativeHandle()));

                std::size_t freeInternal(0u);
                std::size_t totalInternal(0u);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(MemGetInfo)(&freeInternal, &totalInternal));

                return totalInternal;
            }
        };

        //! The CUDA/HIP RT device free memory get trait specialization.
        template<>
        struct GetFreeMemBytes<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getFreeMemBytes(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt const& dev) -> std::size_t
            {
                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.getNativeHandle()));

                std::size_t freeInternal(0u);
                std::size_t totalInternal(0u);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(MemGetInfo)(&freeInternal, &totalInternal));

                return freeInternal;
            }
        };

        //! The CUDA/HIP RT device warp size get trait specialization.
        template<>
        struct GetWarpSizes<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getWarpSizes(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt const& dev) -> std::vector<std::size_t>
            {
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                cudaDeviceProp devProp;
#    endif
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                hipDeviceProp_t devProp;
#    endif
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(GetDeviceProperties)(&devProp, dev.getNativeHandle()));

                return {static_cast<std::size_t>(devProp.warpSize)};
            }
        };

        //! The CUDA/HIP RT device reset trait specialization.
        template<>
        struct Reset<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto reset(ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt const& dev)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(DeviceReset)());
            }
        };

        //! The CUDA/HIP RT device native handle trait specialization.
        template<>
        struct NativeHandle<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            [[nodiscard]] static auto getNativeHandle(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt const& dev)
            {
                return dev.getNativeHandle();
            }
        };

        //! The CUDA/HIP RT device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt, TElem, TDim, TIdx>
        {
            using type = ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::BufUniformCudaHipRt<TElem, TDim, TIdx>;
        };

        //! The CUDA/HIP RT device platform type trait specialization.
        template<>
        struct PltfType<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            using type = ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::PltfUniformCudaHipRt;
        };

        //! The thread CUDA/HIP device wait specialization.
        //!
        //! Blocks until the device has completed all preceding requested tasks.
        //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
        template<>
        struct CurrentThreadWaitFor<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt const& dev) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(DeviceSynchronize)());
            }
        };

        template<>
        struct QueueType<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt, Blocking>
        {
            using type = ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::QueueUniformCudaHipRtBlocking;
        };

        template<>
        struct QueueType<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt, NonBlocking>
        {
            using type = ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::QueueUniformCudaHipRtNonBlocking;
        };
    } // namespace trait
} // namespace alpaka

#endif
