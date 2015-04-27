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

#include <alpaka/host/SysInfo.hpp>      // host::getCpuName, host::getGlobalMemSizeBytes

#include <alpaka/traits/Acc.hpp>        // AccType
#include <alpaka/traits/Dev.hpp>        // DevType
#include <alpaka/traits/Stream.hpp>     // StreamType
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <stdexcept>                    // std::runtime_error
#include <sstream>                      // std::stringstream
#include <limits>                       // std::numeric_limits
#include <thread>                       // std::thread

namespace alpaka
{
    namespace accs
    {
        namespace threads
        {
            namespace detail
            {
                class AccThreads;
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
                    ALPAKA_FCT_HOST auto operator=(DevThreads const &) -> DevThreads & = default;
                    //-----------------------------------------------------------------------------
                    //! Equality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator==(DevThreads const &) const
                    -> bool
                    {
                        return true;
                    }
                    //-----------------------------------------------------------------------------
                    //! Inequality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator!=(DevThreads const & rhs) const
                    -> bool
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
                    -> DevThreads
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
        }
    }

    namespace host
    {
        namespace detail
        {
            template<
                typename TDev>
            class StreamHost;
        }
    }

    namespace accs
    {
        namespace threads
        {
            namespace detail
            {
                using StreamThreads = host::detail::StreamHost<DevThreads>;
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The threads accelerator device accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::threads::detail::DevThreads>
            {
                using type = accs::threads::detail::AccThreads;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The threads accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::DevThreads;
            };
            //#############################################################################
            //! The threads accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::threads::detail::DevManThreads>
            {
                using type = accs::threads::detail::DevThreads;
            };

            //#############################################################################
            //! The threads accelerator device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                accs::threads::detail::DevThreads>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    accs::threads::detail::DevThreads const & dev)
                -> accs::threads::detail::DevThreads
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The threads accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetProps<
                accs::threads::detail::DevThreads>
            {
                ALPAKA_FCT_HOST static auto getProps(
                    accs::threads::detail::DevThreads const &)
                -> alpaka::dev::DevProps
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
                        Vec3<>(
                            uiBlockThreadsCountMax,
                            uiBlockThreadsCountMax,
                            uiBlockThreadsCountMax),
                        // m_v3uiGridBlockExtentsMax
                        Vec3<>(
                            std::numeric_limits<UInt>::max(),
                            std::numeric_limits<UInt>::max(),
                            std::numeric_limits<UInt>::max()),
                        // m_uiGlobalMemSizeBytes
                        host::getGlobalMemSizeBytes());
                }
            };

            //#############################################################################
            //! The threads accelerator device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                accs::threads::detail::DevThreads>
            {
                ALPAKA_FCT_HOST static auto getFreeMemBytes(
                    accs::threads::detail::DevThreads const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    // \FIXME: Get correct free memory size!
                    return host::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The threads accelerator device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                accs::threads::detail::DevThreads>
            {
                ALPAKA_FCT_HOST static auto reset(
                    accs::threads::detail::DevThreads const & dev)
                -> void
                {
                    boost::ignore_unused(dev);

                    // The threads device can not be reset for now.
                }
            };

            //#############################################################################
            //! The threads accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::DevManThreads;
            };
            //#############################################################################
            //! The threads accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::threads::detail::DevThreads>
            {
                using type = accs::threads::detail::DevManThreads;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The threads accelerator device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::threads::detail::DevThreads>
            {
                using type = accs::threads::detail::StreamThreads;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The threads accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                accs::threads::detail::DevThreads>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    accs::threads::detail::DevThreads const &)
                -> void
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
