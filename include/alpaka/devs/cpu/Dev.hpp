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

#include <alpaka/devs/cpu/SysInfo.hpp>  // getCpuName, getGlobalMemSizeBytes

#include <alpaka/traits/Dev.hpp>        // DevType
#include <alpaka/traits/Event.hpp>      // EventType
#include <alpaka/traits/Stream.hpp>     // StreamType
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <sstream>                      // std::stringstream
#include <limits>                       // std::numeric_limits
#include <thread>                       // std::thread

namespace alpaka
{
    namespace devs
    {
        //-----------------------------------------------------------------------------
        //! The CPU device.
        //-----------------------------------------------------------------------------
        namespace cpu
        {
            class DevManCpu;

            //#############################################################################
            //! The CPU device handle.
            //#############################################################################
            class DevCpu
            {
                friend class DevManCpu;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCpu() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCpu(DevCpu const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCpu(DevCpu &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(DevCpu const &) -> DevCpu & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(DevCpu const &) const
                -> bool
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(DevCpu const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }
            };

            //#############################################################################
            //! The CPU device manager.
            //#############################################################################
            class DevManCpu
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevManCpu() = delete;

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
                -> DevCpu
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
            };

            //-----------------------------------------------------------------------------
            //! \return The device this object is bound to.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto getDev()
            -> DevCpu
            {
                return DevManCpu::getDevByIdx(0);
            }
        }
    }

    namespace devs
    {
        namespace cpu
        {
            class EventCpu;
            class StreamCpu;
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The CPU device device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                devs::cpu::DevCpu>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                devs::cpu::DevManCpu>
            {
                using type = devs::cpu::DevCpu;
            };

            //#############################################################################
            //! The CPU device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cpu::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cpu::DevCpu const & dev)
                -> devs::cpu::DevCpu
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The CPU device name get trait specialization.
            //#############################################################################
            template<>
            struct GetName<
                devs::cpu::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getName(
                    devs::cpu::DevCpu const & dev)
                -> std::string
                {
                    boost::ignore_unused(dev);

                    return devs::cpu::detail::getCpuName();
                }
            };

            //#############################################################################
            //! The CPU device available memory get trait specialization.
            //#############################################################################
            template<>
            struct GetMemBytes<
                devs::cpu::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getMemBytes(
                    devs::cpu::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    return devs::cpu::detail::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                devs::cpu::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getFreeMemBytes(
                    devs::cpu::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    // \FIXME: Get correct free memory size!
                    return devs::cpu::detail::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                devs::cpu::DevCpu>
            {
                ALPAKA_FCT_HOST static auto reset(
                    devs::cpu::DevCpu const & dev)
                -> void
                {
                    boost::ignore_unused(dev);

                    // The CPU does nothing on reset.
                }
            };

            //#############################################################################
            //! The CPU device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                devs::cpu::DevCpu>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CPU device event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                devs::cpu::DevCpu>
            {
                using type = devs::cpu::EventCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CPU device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                devs::cpu::DevCpu>
            {
                using type = devs::cpu::StreamCpu;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CPU device thread wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cpu::DevCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::DevCpu const &)
                -> void
                {
                    // Because CPU calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
