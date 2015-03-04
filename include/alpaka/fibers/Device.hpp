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

#include <alpaka/fibers/AccFibersFwd.hpp>   // AccFibers

#include <alpaka/host/SysInfo.hpp>          // host::getCpuName, host::getGlobalMemSizeBytes

#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor
#include <alpaka/traits/Device.hpp>         // DevType

#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits
#include <thread>                           // std::thread

namespace alpaka
{
    namespace fibers
    {
        namespace detail
        {
            // Forward declaration.
            class DeviceManagerFibers;

            //#############################################################################
            //! The fibers accelerator device handle.
            //#############################################################################
            class DeviceFibers
            {
                friend class DeviceManagerFibers;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceFibers() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceFibers(DeviceFibers const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceFibers(DeviceFibers &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceFibers & operator=(DeviceFibers const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(DeviceFibers const &) const
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(DeviceFibers const & rhs) const
                {
                    return !((*this) == rhs);
                }
            };

            //#############################################################################
            //! The fibers accelerator device manager.
            //#############################################################################
            class DeviceManagerFibers
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceManagerFibers() = delete;

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
                ALPAKA_FCT_HOST static fibers::detail::DeviceFibers getDeviceByIdx(
                    std::size_t const & uiIdx)
                {
                    std::size_t const uiNumDevices(getDeviceCount());
                    if(uiIdx >= uiNumDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << uiIdx << " because there are only " << uiNumDevices << " threads devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {};
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the currently used device.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static fibers::detail::DeviceFibers getCurrentDevice()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(
                    fibers::detail::DeviceFibers const & )
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
            //! The fibers accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                AccFibers>
            {
                using type = fibers::detail::DeviceFibers;
            };
            //#############################################################################
            //! The fibers accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                fibers::detail::DeviceManagerFibers>
            {
                using type = fibers::detail::DeviceFibers;
            };

            //#############################################################################
            //! The fibers accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetDevProps<
                fibers::detail::DeviceFibers>
            {
                ALPAKA_FCT_HOST static alpaka::dev::DevProps getDevProps(
                    fibers::detail::DeviceFibers const &)
                {
#if ALPAKA_INTEGRATION_TEST
                    UInt const uiBlockThreadsCountMax(24u);
#else
                    UInt const uiBlockThreadsCountMax(32u); // \TODO: What is the maximum? Just set a reasonable value?
#endif
                    return alpaka::dev::DevProps(
                        // m_sName
                        host::getCpuName(),
                        // m_uiMultiProcessorCount
                        std::thread::hardware_concurrency(), // \TODO: This may be inaccurate.
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_v3uiBlockThreadExtentsMax
                        Vec<3u>(uiBlockThreadsCountMax, uiBlockThreadsCountMax, uiBlockThreadsCountMax),
                        // m_v3uiGridBlockExtentsMax
                        Vec<3u>(std::numeric_limits<UInt>::max(), std::numeric_limits<UInt>::max(), std::numeric_limits<UInt>::max()),
                        // m_uiGlobalMemSizeBytes
                        host::getGlobalMemSizeBytes());
                }
            };

            //#############################################################################
            //! The fibers accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                AccFibers>
            {
                using type = fibers::detail::DeviceManagerFibers;
            };
            //#############################################################################
            //! The fibers accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                fibers::detail::DeviceFibers>
            {
                using type = fibers::detail::DeviceManagerFibers;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The fibers accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                fibers::detail::DeviceFibers>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    fibers::detail::DeviceFibers const &)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
