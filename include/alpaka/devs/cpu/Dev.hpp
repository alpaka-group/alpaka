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
        namespace cpu
        {
            namespace detail
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
            }

            //-----------------------------------------------------------------------------
            //! \return The device this object is bound to.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto getDev()
            -> detail::DevCpu
            {
                return detail::DevManCpu::getDevByIdx(0);
            }
        }
    }

    namespace devs
    {
        namespace cpu
        {
            namespace detail
            {
                class StreamCpu;
            }
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The cpu device device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                devs::cpu::detail::DevCpu>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The cpu device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                devs::cpu::detail::DevManCpu>
            {
                using type = devs::cpu::detail::DevCpu;
            };

            //#############################################################################
            //! The cpu device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cpu::detail::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cpu::detail::DevCpu const & dev)
                -> devs::cpu::detail::DevCpu
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The cpu device name get trait specialization.
            //#############################################################################
            template<>
            struct GetName<
                devs::cpu::detail::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getName(
                    devs::cpu::detail::DevCpu const & dev)
                -> std::string
                {
                    boost::ignore_unused(dev);

                    return devs::cpu::detail::getCpuName();
                }
            };

            //#############################################################################
            //! The cpu device available memory get trait specialization.
            //#############################################################################
            template<>
            struct GetMemBytes<
                devs::cpu::detail::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getMemBytes(
                    devs::cpu::detail::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    return devs::cpu::detail::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The cpu device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                devs::cpu::detail::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getFreeMemBytes(
                    devs::cpu::detail::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    // \FIXME: Get correct free memory size!
                    return devs::cpu::detail::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The cpu device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                devs::cpu::detail::DevCpu>
            {
                ALPAKA_FCT_HOST static auto reset(
                    devs::cpu::detail::DevCpu const & dev)
                -> void
                {
                    boost::ignore_unused(dev);

                    // The fibers device can not be reset for now.
                }
            };

            //#############################################################################
            //! The cpu device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                devs::cpu::detail::DevCpu>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The cpu device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                devs::cpu::detail::DevCpu>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The cpu thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cpu::detail::DevCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::detail::DevCpu const &)
                -> void
                {
                    // Because cpu calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
