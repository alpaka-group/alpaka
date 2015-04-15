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

#include <alpaka/host/SysInfo.hpp>              // host::getCpuName, host::getGlobalMemSizeBytes

#include <alpaka/traits/Dev.hpp>                // DevType
#include <alpaka/traits/Stream.hpp>             // StreamType
#include <alpaka/traits/Wait.hpp>               // CurrentThreadWaitFor

#include <stdexcept>                            // std::runtime_error
#include <sstream>                              // std::stringstream
#include <limits>                               // std::numeric_limits

namespace alpaka
{
    namespace accs
    {
        namespace serial
        {
            namespace detail
            {
                class AccSerial;
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
                    -> DevSerial
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
        namespace serial
        {
            namespace detail
            {
                using StreamSerial = host::detail::StreamHost<DevSerial>;
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The serial accelerator device accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::serial::detail::DevSerial>
            {
                using type = accs::serial::detail::AccSerial;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The serial accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::DevSerial;
            };
            //#############################################################################
            //! The serial accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::serial::detail::DevManSerial>
            {
                using type = accs::serial::detail::DevSerial;
            };

            //#############################################################################
            //! The serial accelerator device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                accs::serial::detail::DevSerial>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    accs::serial::detail::DevSerial const & dev)
                -> accs::serial::detail::DevSerial
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The serial accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetDevProps<
                accs::serial::detail::DevSerial>
            {
                ALPAKA_FCT_HOST static auto getDevProps(
                    accs::serial::detail::DevSerial const &)
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
                        Vec<3u>::ones(),
                        // m_v3uiGridBlockExtentsMax
                        Vec<3u>::all(std::numeric_limits<Vec<3u>::Val>::max()),
                        // m_uiGlobalMemSizeBytes
                        host::getGlobalMemSizeBytes());
                }
            };

            //#############################################################################
            //! The serial accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::DevManSerial;
            };
            //#############################################################################
            //! The serial accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::serial::detail::DevSerial>
            {
                using type = accs::serial::detail::DevManSerial;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The serial accelerator device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::serial::detail::DevSerial>
            {
                using type = accs::serial::detail::StreamSerial;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The serial accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                accs::serial::detail::DevSerial>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    accs::serial::detail::DevSerial const &)
                -> void
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
