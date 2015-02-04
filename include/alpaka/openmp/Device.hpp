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

#include <alpaka/openmp/AccOpenMpFwd.hpp>   // AccOpenMp

#include <alpaka/host/SysInfo.hpp>          // host::getCpuName, host::getGlobalMemSizeBytes

#include <alpaka/openmp/Common.hpp>

#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor
#include <alpaka/traits/Device.hpp>         // GetDev

#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits
#include <thread>                           // std::thread

namespace alpaka
{
    namespace openmp
    {
        namespace detail
        {
            // Forward declaration.
            class DeviceManagerOpenMp;

            //#############################################################################
            //! The OpenMP accelerator device handle.
            //#############################################################################
            class DeviceOpenMp
            {
                friend class DeviceManagerOpenMp;
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceOpenMp() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceOpenMp(DeviceOpenMp const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceOpenMp(DeviceOpenMp &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceOpenMp & operator=(DeviceOpenMp const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(DeviceOpenMp const &) const
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(DeviceOpenMp const & rhs) const
                {
                    return !((*this) == rhs);
                }

            /*private:
                int m_iDevice;*/
            };

            //#############################################################################
            //! The OpenMP accelerator device manager.
            // \TODO: Add ability to offload to Xeon Phi.
            //#############################################################################
            class DeviceManagerOpenMp
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceManagerOpenMp() = delete;

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
                ALPAKA_FCT_HOST static openmp::detail::DeviceOpenMp getDeviceByIdx(
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
                ALPAKA_FCT_HOST static openmp::detail::DeviceOpenMp getCurrentDevice()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(
                    openmp::detail::DeviceOpenMp const & )
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
            //! The OpenMP accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                AccOpenMp>
            {
                using type = openmp::detail::DeviceOpenMp;
            };
            //#############################################################################
            //! The OpenMP accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                openmp::detail::DeviceManagerOpenMp>
            {
                using type = openmp::detail::DeviceOpenMp;
            };

            //#############################################################################
            //! The OpenMP accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetDevProps<
                openmp::detail::DeviceOpenMp>
            {
                ALPAKA_FCT_HOST static alpaka::dev::DevProps getDevProps(
                    openmp::detail::DeviceOpenMp const &)
                {
                    alpaka::dev::DevProps devProps;

                    devProps.m_sName = host::getCpuName();
                    devProps.m_uiMultiProcessorCount = 1u;
#ifdef ALPAKA_INTEGRATION_TEST
                    devProps.m_uiBlockKernelsCountMax = 4u;
#else
                    // HACK: ::omp_get_max_threads() does not return the real limit of the underlying OpenMP runtime:
                    // 'The omp_get_max_threads routine returns the value of the internal control variable, which is used to determine the number of threads that would form the new team, 
                    // if an active parallel region without a num_threads clause were to be encountered at that point in the program.'
                    // How to do this correctly? Is there even a way to get the hard limit apart from omp_set_num_threads(high_value) -> omp_get_max_threads()?
                    ::omp_set_num_threads(1024);
                    devProps.m_uiBlockKernelsCountMax = ::omp_get_max_threads();
#endif
                    devProps.m_v3uiBlockKernelsExtentsMax = Vec<3u>(static_cast<Vec<3u>::Value>(devProps.m_uiBlockKernelsCountMax), static_cast<Vec<3u>::Value>(devProps.m_uiBlockKernelsCountMax), static_cast<Vec<3u>::Value>(devProps.m_uiBlockKernelsCountMax));
                    devProps.m_v3uiGridBlocksExtentsMax = Vec<3u>(std::numeric_limits<Vec<3u>::Value>::max(), std::numeric_limits<Vec<3u>::Value>::max(), std::numeric_limits<Vec<3u>::Value>::max());
                    devProps.m_uiGlobalMemSizeBytes = host::getGlobalMemSizeBytes();
                    //devProps.m_uiMaxClockFrequencyHz = TODO;

                    return devProps;
                }
            };

            //#############################################################################
            //! The OpenMP accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct GetDevMan<
                AccOpenMp>
            {
                using type = openmp::detail::DeviceManagerOpenMp;
            };
            //#############################################################################
            //! The OpenMP accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct GetDevMan<
                openmp::detail::DeviceOpenMp>
            {
                using type = openmp::detail::DeviceManagerOpenMp;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The OpenMP accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                openmp::detail::DeviceOpenMp>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    openmp::detail::DeviceOpenMp const &)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
