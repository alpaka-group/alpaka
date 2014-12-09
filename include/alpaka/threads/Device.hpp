/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/threads/AccThreadsFwd.hpp> // AccThreads

#include <alpaka/host/SystemInfo.hpp>       // host::getCpuName, host::getGlobalMemorySizeBytes

#include <alpaka/interfaces/Device.hpp>     // alpaka::device::DeviceHandle, alpaka::device::DeviceManager

#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits
#include <thread>                           // std::thread

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            // Forward declaration.
            class DeviceManagerThreads;

            //#############################################################################
            //! The CUDA accelerator device handle.
            //#############################################################################
            class DeviceHandleThreads
            {
                friend class DeviceManagerThreads;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleThreads() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleThreads(DeviceHandleThreads const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleThreads(DeviceHandleThreads &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleThreads & operator=(DeviceHandleThreads const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~DeviceHandleThreads() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The device properties.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST device::DeviceProperties getProperties() const
                {
                    device::DeviceProperties deviceProperties;

                    deviceProperties.m_sName = host::getCpuName();
                    // TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
                    // std::thread::hardware_concurrency  can return 0, so a default for this case?
                    deviceProperties.m_uiBlockKernelSizeMax = std::thread::hardware_concurrency() * 128;
                    deviceProperties.m_v3uiBlockKernelSizePerDimMax = vec<3u>(deviceProperties.m_uiBlockKernelSizeMax, deviceProperties.m_uiBlockKernelSizeMax, deviceProperties.m_uiBlockKernelSizeMax);
                    deviceProperties.m_v3uiGridBlockSizePerDimMax = vec<3u>(std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max());
                    deviceProperties.m_uiStreamCount = std::numeric_limits<std::size_t>::max();
                    deviceProperties.m_uiExecutionUnitCount = std::thread::hardware_concurrency();          // TODO: This may be inaccurate.
                    deviceProperties.m_uiGlobalMemorySizeBytes = host::getGlobalMemorySizeBytes();
                    //deviceProperties.m_uiClockFrequencyHz = TODO;

                    return deviceProperties;
                }
            };
        }
    }

    namespace device
    {
        //#############################################################################
        //! The CUDA accelerator device handle.
        //#############################################################################
        template<>
        class DeviceHandle<AccThreads> :
            public device::detail::IDeviceHandle<cuda::detail::DeviceHandleThreads>
        {
            friend class cuda::detail::DeviceManagerThreads;
        private:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle() = default;

        public:
            //-----------------------------------------------------------------------------
            //! Copy-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle(DeviceHandle const &) = default;
            //-----------------------------------------------------------------------------
            //! Move-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle(DeviceHandle &&) = default;
            //-----------------------------------------------------------------------------
            //! Assignment-operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle & operator=(DeviceHandle const &) = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~DeviceHandle() noexcept = default;
        };
    }

    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator device manager.
            //#############################################################################
            class DeviceManagerThreads
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceManagerThreads() = delete;

                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDeviceCount()
                {
                    return 1;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static device::DeviceHandle<AccThreads> getDeviceHandleByIndex(std::size_t const & uiIndex)
                {
                    std::size_t const uiNumDevices(getDeviceCount());
                    if(uiIndex >= uiNumDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << uiIndex << " because there are only " << uiNumDevices << " threads devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {};
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the currently used device.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static device::DeviceHandle<AccThreads> getCurrentDeviceHandle()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(device::DeviceHandle<AccThreads> const & )
                {
                    // The code is already running on this device.
                }
            };
        }
    }

    namespace device
    {
        //#############################################################################
        //! The threads accelerator device manager.
        //#############################################################################
        template<>
        class DeviceManager<AccThreads> :
            public detail::IDeviceManager<cuda::detail::DeviceManagerThreads>
        {};
    }
}
