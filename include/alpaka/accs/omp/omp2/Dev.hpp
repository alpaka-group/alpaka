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

#include <alpaka/accs/omp/Common.hpp>

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
        namespace omp
        {
            namespace omp2
            {
                namespace detail
                {
                    class AccOmp2;
                    class DevManOmp2;

                    //#############################################################################
                    //! The OpenMP2 accelerator device handle.
                    //#############################################################################
                    class DevOmp2
                    {
                        friend class DevManOmp2;
                    protected:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST DevOmp2() = default;
                    public:
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST DevOmp2(DevOmp2 const &) = default;
    #if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST DevOmp2(DevOmp2 &&) = default;
    #endif
                        //-----------------------------------------------------------------------------
                        //! Assignment operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator=(DevOmp2 const &) -> DevOmp2 & = default;
                        //-----------------------------------------------------------------------------
                        //! Equality comparison operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator==(DevOmp2 const &) const
                        -> bool
                        {
                            return true;
                        }
                        //-----------------------------------------------------------------------------
                        //! Inequality comparison operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator!=(DevOmp2 const & rhs) const
                        -> bool
                        {
                            return !((*this) == rhs);
                        }

                    /*private:
                        int m_iDevice;*/
                    };

                    //#############################################################################
                    //! The OpenMP2 accelerator device manager.
                    // \TODO: Add ability to offload to Xeon Phi.
                    //#############################################################################
                    class DevManOmp2
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST DevManOmp2() = delete;

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
                        -> DevOmp2
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
        namespace omp
        {
            namespace omp2
            {
                namespace detail
                {
                    using StreamOmp2 = host::detail::StreamHost<DevOmp2>;
                }
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The OpenMP2 accelerator device accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::omp::omp2::detail::DevOmp2>
            {
                using type = accs::omp::omp2::detail::AccOmp2;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The OpenMP2 accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::DevOmp2;
            };
            //#############################################################################
            //! The OpenMP2 accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::omp::omp2::detail::DevManOmp2>
            {
                using type = accs::omp::omp2::detail::DevOmp2;
            };

            //#############################################################################
            //! The OpenMP2 accelerator device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                accs::omp::omp2::detail::DevOmp2>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    accs::omp::omp2::detail::DevOmp2 const & dev)
                -> accs::omp::omp2::detail::DevOmp2
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The OpenMP2 accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetProps<
                accs::omp::omp2::detail::DevOmp2>
            {
                ALPAKA_FCT_HOST static auto getProps(
                    accs::omp::omp2::detail::DevOmp2 const &)
                -> alpaka::dev::DevProps
                {
#if ALPAKA_INTEGRATION_TEST

                    UInt const uiBlockThreadsCountMax(4u);
#else
                    // m_uiBlockThreadsCountMax
                    // HACK: ::omp_get_max_threads() does not return the real limit of the underlying OpenMP2 runtime:
                    // 'The omp_get_max_threads routine returns the value of the internal control variable, which is used to determine the number of threads that would form the new team,
                    // if an active parallel region without a num_threads clause were to be encountered at that point in the program.'
                    // How to do this correctly? Is there even a way to get the hard limit apart from omp_set_num_threads(high_value) -> omp_get_max_threads()?
                    ::omp_set_num_threads(1024);
                    UInt const uiBlockThreadsCountMax(static_cast<UInt>(::omp_get_max_threads()));
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
            //! The OpenMP2 accelerator device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                accs::omp::omp2::detail::DevOmp2>
            {
                ALPAKA_FCT_HOST static auto getFreeMemBytes(
                    accs::omp::omp2::detail::DevOmp2 const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    // \FIXME: Get correct free memory size!
                    return host::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The OpenMP2 accelerator device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                accs::omp::omp2::detail::DevOmp2>
            {
                ALPAKA_FCT_HOST static auto reset(
                    accs::omp::omp2::detail::DevOmp2 const & dev)
                -> void
                {
                    boost::ignore_unused(dev);

                    // The OpenMP2 device can not be reset for now.
                }
            };

            //#############################################################################
            //! The OpenMP2 accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::DevManOmp2;
            };
            //#############################################################################
            //! The OpenMP2 accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::omp::omp2::detail::DevOmp2>
            {
                using type = accs::omp::omp2::detail::DevManOmp2;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The OpenMP2 accelerator device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::omp::omp2::detail::DevOmp2>
            {
                using type = accs::omp::omp2::detail::StreamOmp2;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The OpenMP2 accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                accs::omp::omp2::detail::DevOmp2>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    accs::omp::omp2::detail::DevOmp2 const &)
                -> void
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}
