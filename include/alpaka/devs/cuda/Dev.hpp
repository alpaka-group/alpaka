/**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Rene Widera
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

#include <alpaka/traits/Acc.hpp>        // AccType
#include <alpaka/traits/Dev.hpp>        // DevType
#include <alpaka/traits/Event.hpp>      // EventType
#include <alpaka/traits/Stream.hpp>     // StreamType
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor

#include <alpaka/core/Cuda.hpp>

#include <iostream>                     // std::cout
#include <sstream>                      // std::stringstream
#include <limits>                       // std::numeric_limits
#include <stdexcept>                    // std::runtime_error

namespace alpaka
{
    namespace devs
    {
        //-----------------------------------------------------------------------------
        //! The CUDA device.
        //-----------------------------------------------------------------------------
        namespace cuda
        {
            template<
                typename TDim>
            class AccGpuCuda;
            class DevManCuda;

            //#############################################################################
            //! The CUDA device handle.
            //#############################################################################
            class DevCuda
            {
                friend class DevManCuda;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCuda() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCuda(DevCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCuda(DevCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(DevCuda const &) -> DevCuda & = default;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(DevCuda &&) -> DevCuda & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(DevCuda const & rhs) const
                -> bool
                {
                    return m_iDevice == rhs.m_iDevice;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(DevCuda const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }

            public:
                int m_iDevice;
            };

            //#############################################################################
            //! The CUDA device manager.
            //#############################################################################
            class DevManCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevManCuda() = delete;

                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getDevCount()
                -> std::size_t
                {
                    int iNumDevices(0);
                    ALPAKA_CUDA_RT_CHECK(cudaGetDeviceCount(&iNumDevices));

                    return static_cast<std::size_t>(iNumDevices);
                }
                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getDevByIdx(
                    std::size_t const & uiIdx)
                -> DevCuda
                {
                    DevCuda dev;

                    std::size_t const uiNumDevices(getDevCount());
                    if(uiIdx >= uiNumDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << uiIdx << " because there are only " << uiNumDevices << " CUDA devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    // Initialize the cuda runtime.
                    //init();

                    // Try all devices if the given one is unusable.
                    for(std::size_t iDeviceOffset(0); iDeviceOffset < uiNumDevices; ++iDeviceOffset)
                    {
                        std::size_t const iDevice((uiIdx + iDeviceOffset) % uiNumDevices);

                        if(isDevUsable(iDevice))
                        {
                            dev.m_iDevice = static_cast<int>(iDevice);

                            // Log this device.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                            cudaDeviceProp devProp;
                            ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(&devProp, dev.m_iDevice));
#endif
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDeviceProperties(devProp);
#elif ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                            std::cout << BOOST_CURRENT_FUNCTION << devProp.name << std::endl;
#endif
                            return dev;
                        }
                    }

                    // If we came until here, none of the devices was usable.
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for device " << uiIdx << " because none of the " << uiNumDevices << " CUDA devices is usable!";
                    throw std::runtime_error(ssErr.str());
                }

            private:
                //-----------------------------------------------------------------------------
                //! \return If the device is usable.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto isDevUsable(
                    std::size_t iDevice)
                -> bool
                {
                    cudaError rc(cudaSetDevice(static_cast<int>(iDevice)));

                    // Create a dummy stream to check if the device is already used by an other process.
                    // \TODO: Check if this workaround is needed!
                    // Since NVIDIA changed something in the runtime cudaSetDevice never returns an error if another process already uses the selected device and gpu compute mode is set "process exclusive".
                    if(rc == cudaSuccess)
                    {
                        cudaStream_t stream;
                        rc = cudaStreamCreate(&stream);
                    }

                    if(rc == cudaSuccess)
                    {
                        return true;
                    }
                    else
                    {
                        ALPAKA_CUDA_RT_CHECK(rc);
                        // Reset the Error state.
                        cudaGetLastError();

                        return false;
                    }
                }
                //-----------------------------------------------------------------------------
                //! Initializes the cuda runtime.
                //-----------------------------------------------------------------------------
                /*ALPAKA_FCT_HOST static auto init()
                -> void
                {
                    static bool s_bInitialized = false;

                    if(!s_bInitialized)
                    {
                        s_bInitialized = true;

                        // - cudaDeviceScheduleSpin:
                        //   Instruct CUDA to actively spin when waiting for results from the device.
                        //   This can decrease latency when waiting for the device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.
                        // - cudaDeviceMapHost:
                        //   This flag must be set in order to allocate pinned host memory that is accessible to the device.
                        //   If this flag is not set, cudaHostGetDevicePointer() will always return a failure code.
                        // NOTE: This is disabled because we have to support interop with native CUDA applications.
                        // They could already have set a device before calling into alpaka and we would get:
                        // cudaSetDeviceFlags( 0x01 | 0x08)' returned error: 'cannot set while device is active in this process'
                        //ALPAKA_CUDA_RT_CHECK(cudaSetDeviceFlags(
                        //    cudaDeviceScheduleSpin | cudaDeviceMapHost));
                    }
                }*/

            private:
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                //-----------------------------------------------------------------------------
                //! Prints all the device properties to std::cout.
                //-----------------------------------------------------------------------------
                static auto printDeviceProperties(
                    cudaDeviceProp const & devProp)
                -> void
                {
                    std::size_t const uiKiB(1024);
                    std::size_t const uiMiB(uiKiB * uiKiB);
                    std::cout << "name: " << devProp.name << std::endl;
                    std::cout << "totalGlobalMem: " << devProp.totalGlobalMem/uiMiB << " MiB" << std::endl;
                    std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock/uiKiB << " KiB" << std::endl;
                    std::cout << "regsPerBlock: " << devProp.regsPerBlock << std::endl;
                    std::cout << "warpSize: " << devProp.warpSize << std::endl;
                    std::cout << "memPitch: " << devProp.memPitch << " B" << std::endl;
                    std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
                    std::cout << "maxThreadsDim[3]: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
                    std::cout << "maxGridSize[3]: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")" << std::endl;
                    std::cout << "clockRate: " << devProp.clockRate << " kHz" << std::endl;
                    std::cout << "totalConstMem: " << devProp.totalConstMem/uiKiB << " KiB" << std::endl;
                    std::cout << "major: " << devProp.major << std::endl;
                    std::cout << "minor: " << devProp.minor << std::endl;
                    std::cout << "textureAlignment: " << devProp.textureAlignment << std::endl;
                    std::cout << "texturePitchAlignment: " << devProp.texturePitchAlignment << std::endl;
                    //std::cout << "deviceOverlap: " << devProp.deviceOverlap << std::endl;    // Deprecated
                    std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
                    std::cout << "kernelExecTimeoutEnabled: " << devProp.kernelExecTimeoutEnabled << std::endl;
                    std::cout << "integrated: " << devProp.integrated << std::endl;
                    std::cout << "canMapHostMemory: " << devProp.canMapHostMemory << std::endl;
                    std::cout << "computeMode: " << devProp.computeMode << std::endl;
                    std::cout << "maxTexture1D: " << devProp.maxTexture1D << std::endl;
                    std::cout << "maxTexture1DLinear: " << devProp.maxTexture1DLinear << std::endl;
                    std::cout << "maxTexture2D[2]: " << devProp.maxTexture2D[0] << "x" << devProp.maxTexture2D[1] << std::endl;
                    std::cout << "maxTexture2DLinear[3]: " << devProp.maxTexture2DLinear[0] << "x" << devProp.maxTexture2DLinear[1] << "x" << devProp.maxTexture2DLinear[2] << std::endl;
                    std::cout << "maxTexture2DGather[2]: " << devProp.maxTexture2DGather[0] << "x" << devProp.maxTexture2DGather[1] << std::endl;
                    std::cout << "maxTexture3D[3]: " << devProp.maxTexture3D[0] << "x" << devProp.maxTexture3D[1] << "x" << devProp.maxTexture3D[2] << std::endl;
                    std::cout << "maxTextureCubemap: " << devProp.maxTextureCubemap << std::endl;
                    std::cout << "maxTexture1DLayered[2]: " << devProp.maxTexture1DLayered[0] << "x" << devProp.maxTexture1DLayered[1] << std::endl;
                    std::cout << "maxTexture2DLayered[3]: " << devProp.maxTexture2DLayered[0] << "x" << devProp.maxTexture2DLayered[1] << "x" << devProp.maxTexture2DLayered[2] << std::endl;
                    std::cout << "maxTextureCubemapLayered[2]: " << devProp.maxTextureCubemapLayered[0] << "x" << devProp.maxTextureCubemapLayered[1] << std::endl;
                    std::cout << "maxSurface1D: " << devProp.maxSurface1D << std::endl;
                    std::cout << "maxSurface2D[2]: " << devProp.maxSurface2D[0] << "x" << devProp.maxSurface2D[1] << std::endl;
                    std::cout << "maxSurface3D[3]: " << devProp.maxSurface3D[0] << "x" << devProp.maxSurface3D[1] << "x" << devProp.maxSurface3D[2] << std::endl;
                    std::cout << "maxSurface1DLayered[2]: " << devProp.maxSurface1DLayered[0] << "x" << devProp.maxSurface1DLayered[1] << std::endl;
                    std::cout << "maxSurface2DLayered[3]: " << devProp.maxSurface2DLayered[0] << "x" << devProp.maxSurface2DLayered[1] << "x" << devProp.maxSurface2DLayered[2] << std::endl;
                    std::cout << "maxSurfaceCubemap: " << devProp.maxSurfaceCubemap << std::endl;
                    std::cout << "maxSurfaceCubemapLayered[2]: " << devProp.maxSurfaceCubemapLayered[0] << "x" << devProp.maxSurfaceCubemapLayered[1] << std::endl;
                    std::cout << "surfaceAlignment: " << devProp.surfaceAlignment << std::endl;
                    std::cout << "concurrentKernels: " << devProp.concurrentKernels << std::endl;
                    std::cout << "ECCEnabled: " << devProp.ECCEnabled << std::endl;
                    std::cout << "pciBusID: " << devProp.pciBusID << std::endl;
                    std::cout << "pciDeviceID: " << devProp.pciDeviceID << std::endl;
                    std::cout << "pciDomainID: " << devProp.pciDomainID << std::endl;
                    std::cout << "tccDriver: " << devProp.tccDriver << std::endl;
                    std::cout << "asyncEngineCount: " << devProp.asyncEngineCount << std::endl;
                    std::cout << "unifiedAddressing: " << devProp.unifiedAddressing << std::endl;
                    std::cout << "memoryClockRate: " << devProp.memoryClockRate << " kHz" << std::endl;
                    std::cout << "memoryBusWidth: " << devProp.memoryBusWidth << " b" << std::endl;
                    std::cout << "l2CacheSize: " << devProp.l2CacheSize << " B" << std::endl;
                    std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
                }
#endif
            };

            class EventCuda;
            class StreamCuda;
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The CUDA device device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                devs::cuda::DevCuda>
            {
                using type = devs::cuda::DevCuda;
            };
            //#############################################################################
            //! The CUDA device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                devs::cuda::DevManCuda>
            {
                using type = devs::cuda::DevCuda;
            };

            //#############################################################################
            //! The CUDA device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cuda::DevCuda>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cuda::DevCuda const & dev)
                -> devs::cuda::DevCuda
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The CUDA device name get trait specialization.
            //#############################################################################
            template<>
            struct GetName<
                devs::cuda::DevCuda>
            {
                ALPAKA_FCT_HOST static auto getName(
                    devs::cuda::DevCuda const & dev)
                -> std::string
                {
                    cudaDeviceProp cudaDevProp;
                    ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(
                        &cudaDevProp,
                        dev.m_iDevice));

                    return std::string(cudaDevProp.name);
                }
            };

            //#############################################################################
            //! The CUDA device available memory get trait specialization.
            //#############################################################################
            template<>
            struct GetMemBytes<
                devs::cuda::DevCuda>
            {
                ALPAKA_FCT_HOST static auto getMemBytes(
                    devs::cuda::DevCuda const & dev)
                -> std::size_t
                {
                    // \TODO: Check which is faster: cudaMemGetInfo().totalInternal vs cudaGetDeviceProperties().totalGlobalMem
                    // \TODO: This should be secured by a lock.

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_CUDA_RT_CHECK(cudaMemGetInfo(
                        &freeInternal,
                        &totalInternal));

                    return totalInternal;
                }
            };

            //#############################################################################
            //! The CUDA device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                devs::cuda::DevCuda>
            {
                ALPAKA_FCT_HOST static auto getFreeMemBytes(
                    devs::cuda::DevCuda const & dev)
                -> std::size_t
                {
                    // \TODO: This should be secured by a lock.

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_CUDA_RT_CHECK(cudaMemGetInfo(
                        &freeInternal,
                        &totalInternal));

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The CUDA device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                devs::cuda::DevCuda>
            {
                ALPAKA_FCT_HOST static auto reset(
                    devs::cuda::DevCuda const & dev)
                -> void
                {
                    // \TODO: This should be secured by a lock.

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceReset());
                }
            };
            //#############################################################################
            //! The CUDA device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                devs::cuda::DevCuda>
            {
                using type = devs::cuda::DevManCuda;
            };
            //#############################################################################
            //! The CUDA device manager device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                devs::cuda::DevManCuda>
            {
                using type = devs::cuda::DevManCuda;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CUDA device event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                devs::cuda::DevCuda>
            {
                using type = devs::cuda::EventCuda;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CUDA device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                devs::cuda::DevCuda>
            {
                using type = devs::cuda::StreamCuda;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The thread CUDA device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or streams that are created after this call is made are not waited for.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cuda::DevCuda>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cuda::DevCuda const & dev)
                -> void
                {
                    // \TODO: This should be secured by a lock.

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceSynchronize());
                }
            };
        }
    }
}
