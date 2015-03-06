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

#include <alpaka/host/SysInfo.hpp>          // host::getCpuName, host::getGlobalMemSizeBytes

#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor
#include <alpaka/traits/Device.hpp>         // DevType

#include <stdexcept>                        // std::runtime_error
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
            class DevManThreads;

            //#############################################################################
            //! The threads accelerator device handle.
            //#############################################################################
            class DevThreads
            {
                friend class DevManThreads;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevThreads() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevThreads(DevThreads const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevThreads(DevThreads &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevThreads & operator=(DevThreads const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(DevThreads const &) const
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(DevThreads const & rhs) const
                {
                    return !((*this) == rhs);
                }
            };

            //#############################################################################
            //! The threads accelerator device manager.
            //#############################################################################
            class DevManThreads
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevManThreads() = delete;

                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDevCount()
                {
                    return 1;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static threads::detail::DevThreads getDevByIdx(
                    std::size_t const & uiIdx)
                {
                    std::size_t const uiNumDevices(getDevCount());
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
                ALPAKA_FCT_HOST static threads::detail::DevThreads getCurrentDev()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDev(
                    threads::detail::DevThreads const & )
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
            //! The threads accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                AccThreads>
            {
                using type = threads::detail::DevThreads;
            };
            //#############################################################################
            //! The threads accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                threads::detail::DevManThreads>
            {
                using type = threads::detail::DevThreads;
            };

            //#############################################################################
            //! The threads accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetDevProps<
                threads::detail::DevThreads>
            {
                ALPAKA_FCT_HOST static alpaka::dev::DevProps getDevProps(
                    threads::detail::DevThreads const &)
                {
#if ALPAKA_INTEGRATION_TEST
                    UInt const uiBlockThreadsCountMax(8u);
#else
                    // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
                    // std::thread::hardware_concurrency  can return 0, so a default for this case?
                    UInt const uiBlockThreadsCountMax(std::thread::hardware_concurrency() * 8u);
#endif
                    return alpaka::dev::DevProps(
                        // m_sName
                        host::getCpuName(),
                        // m_uiMultiProcessorCount
                        1u,
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
            //! The threads accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                AccThreads>
            {
                using type = threads::detail::DevManThreads;
            };
            //#############################################################################
            //! The threads accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                threads::detail::DevThreads>
            {
                using type = threads::detail::DevManThreads;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The threads accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                threads::detail::DevThreads>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    threads::detail::DevThreads const &)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
