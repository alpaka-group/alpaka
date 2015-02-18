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

#include <alpaka/host/SysInfo.hpp>          // host::getCpuName, host::getGlobalMemSizeBytes

#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor
#include <alpaka/traits/Device.hpp>         // DevType

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
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceSerial(DeviceSerial &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceSerial & operator=(DeviceSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(DeviceSerial const &) const
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
            };

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
                ALPAKA_FCT_HOST static serial::detail::DeviceSerial getDeviceByIdx(
                    std::size_t const & uiIdx)
                {
                    std::size_t const uiNumDevices(getDeviceCount());
                    if(uiIdx >= uiNumDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << uiIdx << " because there are only " << uiNumDevices << " serial devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {};
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the currently used device.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static serial::detail::DeviceSerial getCurrentDevice()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(
                    serial::detail::DeviceSerial const & )
                {
                    // The code is already running on this device.
                }
            };
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The serial accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                AccSerial>
            {
                using type = serial::detail::DeviceSerial;
            };
            //#############################################################################
            //! The serial accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                serial::detail::DeviceManagerSerial>
            {
                using type = serial::detail::DeviceSerial;
            };

            //#############################################################################
            //! The serial accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetDevProps<
                serial::detail::DeviceSerial>
            {
                ALPAKA_FCT_HOST static alpaka::dev::DevProps getDevProps(
                    serial::detail::DeviceSerial const &)
                {
                    alpaka::dev::DevProps devProps;

                    devProps.m_sName = host::getCpuName();
                    devProps.m_uiMultiProcessorCount = 1u;
                    devProps.m_uiBlockKernelsCountMax = 1u;
                    devProps.m_v3uiBlockKernelsExtentsMax = Vec<3u>(1u, 1u, 1u);
                    devProps.m_v3uiGridBlocksExtentsMax = Vec<3u>(std::numeric_limits<Vec<1u>::Val>::max(), std::numeric_limits<Vec<1u>::Val>::max(), std::numeric_limits<Vec<1u>::Val>::max());
                    devProps.m_uiGlobalMemSizeBytes = host::getGlobalMemSizeBytes();
                    //devProps.m_uiMaxClockFrequencyHz = TODO;

                    return devProps;
                }
            };

            //#############################################################################
            //! The serial accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                AccSerial>
            {
                using type = serial::detail::DeviceManagerSerial;
            };
            //#############################################################################
            //! The serial accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                serial::detail::DeviceSerial>
            {
                using type = serial::detail::DeviceManagerSerial;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The serial accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                serial::detail::DeviceSerial>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    serial::detail::DeviceSerial const &)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
