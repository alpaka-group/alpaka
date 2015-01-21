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

#include <alpaka/threads/AccThreadsFwd.hpp> // AccThreads

#include <alpaka/host/SystemInfo.hpp>       // host::getCpuName, host::getGlobalMemorySizeBytes

#include <alpaka/interfaces/Device.hpp>     // alpaka::device::Device, alpaka::device::DeviceManager

#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits
#include <thread>                           // std::thread

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            // Forward declaration.
            class DeviceManagerThreads;

            //#############################################################################
            //! The threads accelerator device handle.
            //#############################################################################
            class DeviceThreads
            {
                friend class DeviceManagerThreads;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceThreads() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceThreads(DeviceThreads const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceThreads(DeviceThreads &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceThreads & operator=(DeviceThreads const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(DeviceThreads const &) const
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(DeviceThreads const & rhs) const
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~DeviceThreads() noexcept = default;

            protected:
                //-----------------------------------------------------------------------------
                //! \return The device properties.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST device::DeviceProperties getProperties() const
                {
                    device::DeviceProperties deviceProperties;

                    deviceProperties.m_sName = host::getCpuName();
                    deviceProperties.m_uiMultiProcessorCount = 1u;
                    // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
                    // std::thread::hardware_concurrency  can return 0, so a default for this case?
                    deviceProperties.m_uiBlockKernelsCountMax = std::thread::hardware_concurrency() * 8u;
                    deviceProperties.m_v3uiBlockKernelsExtentMax = Vec<3u>(deviceProperties.m_uiBlockKernelsCountMax, deviceProperties.m_uiBlockKernelsCountMax, deviceProperties.m_uiBlockKernelsCountMax);
                    deviceProperties.m_v3uiGridBlocksExtentMax = Vec<3u>(std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max());
                    deviceProperties.m_uiGlobalMemorySizeBytes = host::getGlobalMemorySizeBytes();
                    //deviceProperties.m_uiMaxClockFrequencyHz = TODO;

                    return deviceProperties;
                }
            };
        }
    }

    namespace device
    {
        //#############################################################################
        //! The threads accelerator interfaced device handle.
        //#############################################################################
        template<>
        class Device<
            AccThreads> :
            public device::detail::IDevice<threads::detail::DeviceThreads>
        {
            friend class threads::detail::DeviceManagerThreads;
        private:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Device() = default;

        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Device(Device const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Device(Device &&) = default;
            //-----------------------------------------------------------------------------
            //! Assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Device & operator=(Device const &) = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST virtual ~Device() noexcept = default;
        };

        namespace detail
        {
            //#############################################################################
            //! The threads accelerator thread device waiter.
            //#############################################################################
            template<>
            struct ThreadWaitDevice<
                Device<AccThreads>>
            {
                ALPAKA_FCT_HOST ThreadWaitDevice(
                    Device<AccThreads> const &)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }

    namespace threads
    {
        namespace detail
        {
            //#############################################################################
            //! The threads accelerator device manager.
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
                ALPAKA_FCT_HOST static device::Device<AccThreads> getDeviceByIndex(
                    std::size_t const & uiIndex)
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
                ALPAKA_FCT_HOST static device::Device<AccThreads> getCurrentDevice()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(
                    device::Device<AccThreads> const & )
                {
                    // The code is already running on this device.
                }
            };
        }
    }

    namespace device
    {
        //#############################################################################
        //! The threads accelerator interfaced device manager.
        //#############################################################################
        template<>
        class DeviceManager<
            AccThreads> :
            public detail::IDeviceManager<threads::detail::DeviceManagerThreads>
        {};
    }
}
