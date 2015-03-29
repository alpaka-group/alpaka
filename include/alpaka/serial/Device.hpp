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

#include <stdexcept>                        // std::runtime_error
#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            // Forward declaration.
            class DevManSerial;

            //#############################################################################
            //! The serial accelerator device handle.
            //#############################################################################
            class DevSerial
            {
                friend class DevManSerial;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevSerial() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevSerial(DevSerial const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevSerial(DevSerial &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(DevSerial const &) -> DevSerial & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(DevSerial const &) const
                -> bool
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(DevSerial const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }
            };

            //#############################################################################
            //! The serial accelerator device manager.
            //#############################################################################
            class DevManSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevManSerial() = delete;

                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getDevCount()
                -> std::size_t
                {
                    return 1;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getDevByIdx(
                    std::size_t const & uiIdx)
                -> serial::detail::DevSerial
                {
                    std::size_t const uiNumDevices(getDevCount());
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
                ALPAKA_FCT_HOST static auto getCurrentDev()
                -> serial::detail::DevSerial
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto setCurrentDev(
                    serial::detail::DevSerial const & )
                -> void
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
                using type = serial::detail::DevSerial;
            };
            //#############################################################################
            //! The serial accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                serial::detail::DevManSerial>
            {
                using type = serial::detail::DevSerial;
            };

            //#############################################################################
            //! The serial accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetDevProps<
                serial::detail::DevSerial>
            {
                ALPAKA_FCT_HOST static auto getDevProps(
                    serial::detail::DevSerial const &)
                -> alpaka::dev::DevProps
                {
                    return alpaka::dev::DevProps(
                        // m_sName
                        host::getCpuName(),
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        1u,
                        // m_v3uiBlockThreadExtentsMax
                        Vec<3u>(1u, 1u, 1u),
                        // m_v3uiGridBlockExtentsMax
                        Vec<3u>(std::numeric_limits<Vec<1u>::Val>::max(), std::numeric_limits<Vec<1u>::Val>::max(), std::numeric_limits<Vec<1u>::Val>::max()),
                        // m_uiGlobalMemSizeBytes
                        host::getGlobalMemSizeBytes());
                }
            };

            //#############################################################################
            //! The serial accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                AccSerial>
            {
                using type = serial::detail::DevManSerial;
            };
            //#############################################################################
            //! The serial accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                serial::detail::DevSerial>
            {
                using type = serial::detail::DevManSerial;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The serial accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                serial::detail::DevSerial>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    serial::detail::DevSerial const &)
                -> void
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
