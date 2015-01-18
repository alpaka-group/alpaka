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

#include <alpaka/serial/AccSerialFwd.hpp>   // AccSerial

#include <alpaka/host/SystemInfo.hpp>       // host::getCpuName, host::getGlobalMemorySizeBytes

#include <alpaka/interfaces/Device.hpp>     // alpaka::device::Device, alpaka::device::DeviceManager

#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            // Forward declaration.
            class DeviceManagerSerial;

            //#############################################################################
            //! The serial accelerator device handle.
            //#############################################################################
            class DeviceSerial
            {
                friend class DeviceManagerSerial;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceSerial() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceSerial(DeviceSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceSerial(DeviceSerial &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceSerial & operator=(DeviceSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(DeviceSerial const & rhs) const
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(DeviceSerial const & rhs) const
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~DeviceSerial() noexcept = default;

            protected:
                //-----------------------------------------------------------------------------
                //! \return The device properties.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST device::DeviceProperties getProperties() const
                {
                    device::DeviceProperties deviceProperties;

                    deviceProperties.m_sName = host::getCpuName();
                    deviceProperties.m_uiMultiProcessorCount = 1;
                    deviceProperties.m_uiBlockKernelsCountMax = 1;
                    deviceProperties.m_v3uiBlockKernelsExtentMax = Vec<3u>(1u, 1u, 1u);
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
        //! The serial accelerator interfaced device handle.
        //#############################################################################
        template<>
        class Device<
            AccSerial> :
            public device::detail::IDevice<serial::detail::DeviceSerial>
        {
            friend class serial::detail::DeviceManagerSerial;
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
            //! The serial accelerator thread device waiter.
            //#############################################################################
            template<>
            struct ThreadWaitDevice<
                Device<AccSerial>>
            {
                ALPAKA_FCT_HOST ThreadWaitDevice(
                    Device<AccSerial> const & device)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }

    namespace serial
    {
        namespace detail
        {
            //#############################################################################
            //! The serial accelerator device manager.
            //#############################################################################
            class DeviceManagerSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceManagerSerial() = delete;

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
                ALPAKA_FCT_HOST static device::Device<AccSerial> getDeviceByIndex(
                    std::size_t const & uiIndex)
                {
                    std::size_t const uiNumDevices(getDeviceCount());
                    if(uiIndex >= uiNumDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << uiIndex << " because there are only " << uiNumDevices << " serial devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {};
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the currently used device.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static device::Device<AccSerial> getCurrentDevice()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(
                    device::Device<AccSerial> const & )
                {
                    // The code is already running on this device.
                }
            };
        }
    }

    namespace device
    {
        //#############################################################################
        //! The serial accelerator interfaced device manager.
        //#############################################################################
        template<>
        class DeviceManager<
            AccSerial> :
            public detail::IDeviceManager<serial::detail::DeviceManagerSerial>
        {};
    }
}
