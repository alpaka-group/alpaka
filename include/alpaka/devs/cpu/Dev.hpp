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
#include <mutex>                        // std::mutex
#include <memory>                       // std::shared_ptr

namespace alpaka
{
    namespace devs
    {
        //-----------------------------------------------------------------------------
        //! The CPU device.
        //-----------------------------------------------------------------------------
        namespace cpu
        {
            namespace detail
            {
                class StreamCpuImpl;

                //#############################################################################
                //! The CPU device implementation.
                //#############################################################################
                class DevCpuImpl
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCpuImpl() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCpuImpl(DevCpuImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCpuImpl(DevCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(DevCpuImpl const &) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(DevCpuImpl &&) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~DevCpuImpl() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto RegisterStream(std::shared_ptr<StreamCpuImpl> spStreamImpl)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this stream on the device.
                        m_vwpStreams.emplace_back(spStreamImpl);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto UnregisterStream(StreamCpuImpl const * const pStream)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Unregister this stream from the device.
                        m_vwpStreams.erase(
                            std::remove_if(
                                m_vwpStreams.begin(),
                                m_vwpStreams.end(),
                                [pStream](std::weak_ptr<detail::StreamCpuImpl> const & wp)
                                {
                                    return (pStream == wp.lock().get());
                                }),
                            m_vwpStreams.end());
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto GetRegisteredStreams() const
                    -> std::vector<std::shared_ptr<StreamCpuImpl>>
                    {
                        std::vector<std::shared_ptr<devs::cpu::detail::StreamCpuImpl>> vspStreams;

                        std::lock_guard<std::mutex> lk(m_Mutex);

                        for(auto && wpStream : m_vwpStreams)
                        {
                            auto spStream(wpStream.lock());
                            if(spStream)
                            {
                                vspStreams.emplace_back(std::move(spStream));
                            }
                            else
                            {
                                throw std::logic_error("One of the streams registered on the device is invalid!");
                            }
                        }
                        return vspStreams;
                    }

                private:
                    std::mutex mutable m_Mutex;
                    std::vector<std::weak_ptr<StreamCpuImpl>> m_vwpStreams;
                };
            }
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
                ALPAKA_FCT_HOST DevCpu() :
                    m_spDevCpuImpl(std::make_shared<detail::DevCpuImpl>())
                {}
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCpu(DevCpu const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DevCpu(DevCpu &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(DevCpu const &) -> DevCpu & = default;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(DevCpu &&) -> DevCpu & = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~DevCpu() noexcept = default;
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

            public:
                std::shared_ptr<detail::DevCpuImpl> m_spDevCpuImpl;
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
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or streams that are created after this call is made are not waited for.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cpu::DevCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::DevCpu const & dev)
                -> void
                {
                    // Get all the streams on the device at the time of invocation.
                    // All streams added afterwards are ignored.
                    auto vspStreams(
                        dev.m_spDevCpuImpl->GetRegisteredStreams());

                    // Enqueue an event in every stream on the device.
                    // \TODO: This should be done atomically for all streams. 
                    // Furthermore there should not even be a chance to enqueue something between getting the streams and adding our wait events!
                    std::vector<devs::cpu::EventCpu> vEvents;
                    for(auto && spStream : vspStreams)
                    {
                        vEvents.emplace_back(dev);
                        alpaka::stream::enqueue(spStream, vEvents.back());
                    }

                    // Now wait for all the events.
                    for(auto && event : vEvents)
                    {
                        alpaka::wait::wait(event);
                    }
                }
            };
        }
    }
}
