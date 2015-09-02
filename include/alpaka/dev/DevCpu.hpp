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

#include <alpaka/dev/Traits.hpp>        // dev::traits::DevType
#include <alpaka/event/Traits.hpp>      // event::traits::EventType
#include <alpaka/wait/Traits.hpp>       // CurrentThreadWaitFor
#include <alpaka/mem/buf/Traits.hpp>    // mem::buf::traits::BufType
#include <alpaka/mem/view/Traits.hpp>   // mem::view::traits::ViewType

#include <alpaka/stream/Traits.hpp>     // stream::enqueue
#include <alpaka/dev/cpu/SysInfo.hpp>   // getCpuName, getTotalGlobalMemSizeBytes, getFreeGlobalMemSizeBytes

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <map>                          // std::map
#include <sstream>                      // std::stringstream
#include <limits>                       // std::numeric_limits
#include <thread>                       // std::thread
#include <mutex>                        // std::mutex
#include <memory>                       // std::shared_ptr

namespace alpaka
{
    namespace stream
    {
        class StreamCpuAsync;

        namespace cpu
        {
            namespace detail
            {
                class StreamCpuAsyncImpl;
            }
        }
    }
    namespace dev
    {
        //-----------------------------------------------------------------------------
        //! The CPU device.
        //-----------------------------------------------------------------------------
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device implementation.
                //#############################################################################
                class DevCpuImpl
                {
                    friend stream::StreamCpuAsync;                   // stream::StreamCpuAsync::StreamCpuAsync calls RegisterAsyncStream.
                    friend stream::cpu::detail::StreamCpuAsyncImpl;  // StreamCpuAsyncImpl::~StreamCpuAsyncImpl calls UnregisterAsyncStream.
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST DevCpuImpl() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST DevCpuImpl(DevCpuImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST DevCpuImpl(DevCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(DevCpuImpl const &) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(DevCpuImpl &&) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~DevCpuImpl() = default;

                    //-----------------------------------------------------------------------------
                    //! \return The list of all streams on this device.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto GetAllAsyncStreamImpls() const noexcept(false)
                    -> std::vector<std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>>
                    {
                        std::vector<std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>> vspStreams;

                        std::lock_guard<std::mutex> lk(m_Mutex);

                        for(auto const & pairStream : m_mapStreams)
                        {
                            auto spStream(pairStream.second.lock());
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
                    //-----------------------------------------------------------------------------
                    //! Registers the given stream on this device.
                    //! NOTE: Every stream has to be registered for correct functionality of device wait operations!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto RegisterAsyncStream(std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl> spStreamImpl)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this stream on the device.
                        // NOTE: We have to store the plain pointer next to the weak pointer.
                        // This is necessary to find the entry on unregistering because the weak pointer will already be invalid at that point.
                        m_mapStreams.emplace(spStreamImpl.get(), spStreamImpl);
                    }
                    //-----------------------------------------------------------------------------
                    //! Unregisters the given stream from this device.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto UnregisterAsyncStream(stream::cpu::detail::StreamCpuAsyncImpl const * const pStream) noexcept(false)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Unregister this stream from the device.
                        auto const itFind(std::find_if(
                            m_mapStreams.begin(),
                            m_mapStreams.end(),
                            [pStream](std::pair<stream::cpu::detail::StreamCpuAsyncImpl *, std::weak_ptr<stream::cpu::detail::StreamCpuAsyncImpl>> const & pair)
                            {
                                return (pStream == pair.first);
                            }));
                        if(itFind != m_mapStreams.end())
                        {
                            m_mapStreams.erase(itFind);
                        }
                        else
                        {
                            throw std::logic_error("The stream to unregister from the device could not be found in the list of registered streams!");
                        }
                    }

                private:
                    std::mutex mutable m_Mutex;
                    std::map<stream::cpu::detail::StreamCpuAsyncImpl *, std::weak_ptr<stream::cpu::detail::StreamCpuAsyncImpl>> m_mapStreams;
                };
            }
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
            ALPAKA_FN_HOST DevCpu() :
                m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
            {}
        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST DevCpu(DevCpu const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST DevCpu(DevCpu &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(DevCpu const &) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(DevCpu &&) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~DevCpu() = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevCpu const &) const
            -> bool
            {
                return true;
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevCpu const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }

        public:
            std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
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
            ALPAKA_FN_HOST DevManCpu() = delete;

            //-----------------------------------------------------------------------------
            //! \return The number of devices available.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDevCount()
            -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return 1;
            }
            //-----------------------------------------------------------------------------
            //! \return The number of devices available.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDevByIdx(
                std::size_t const & devIdx)
            -> DevCpu
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                std::size_t const devCount(getDevCount());
                if(devIdx >= devCount)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for CPU device with index " << devIdx << " because there are only " << devCount << " devices!";
                    throw std::runtime_error(ssErr.str());
                }

                return {};
            }
        };

        namespace cpu
        {
            //-----------------------------------------------------------------------------
            //! \return The device this object is bound to.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto getDev()
            -> DevCpu
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return DevManCpu::getDevByIdx(0);
            }
        }
    }

    namespace event
    {
        class EventCpu;
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                dev::DevCpu>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                dev::DevManCpu>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The CPU device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    dev::DevCpu const & dev)
                -> dev::DevCpu
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The CPU device name get trait specialization.
            //#############################################################################
            template<>
            struct GetName<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevCpu const & dev)
                -> std::string
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getCpuName();
                }
            };

            //#############################################################################
            //! The CPU device available memory get trait specialization.
            //#############################################################################
            template<>
            struct GetMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getTotalGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getFreeGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    boost::ignore_unused(dev);

                    // The CPU does nothing on reset.
                }
            };

            //#############################################################################
            //! The CPU device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                dev::DevCpu>
            {
                using type = dev::DevManCpu;
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            class BufCpu;

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory buffer type trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct BufType<
                    dev::DevCpu,
                    TElem,
                    TDim,
                    TSize>
                {
                    using type = mem::buf::BufCpu<TElem, TDim, TSize>;
                };
            }
        }
        namespace view
        {
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            class ViewBasic;

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory view type trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct ViewType<
                    dev::DevCpu,
                    TElem,
                    TDim,
                    TSize>
                {
                    using type = mem::view::ViewBasic<dev::DevCpu, TElem, TDim, TSize>;
                };
            }
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device thread wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or streams that are created after this call is made are not waited for.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Get all the streams on the device at the time of invocation.
                    // All streams added afterwards are ignored.
                    auto vspStreams(
                        dev.m_spDevCpuImpl->GetAllAsyncStreamImpls());

                    // Enqueue an event in every asynchronous stream on the device.
                    // \TODO: This should be done atomically for all streams.
                    // Furthermore there should not even be a chance to enqueue something between getting the streams and adding our wait events!
                    std::vector<event::EventCpu> vEvents;
                    for(auto && spStream : vspStreams)
                    {
                        vEvents.emplace_back(dev);
                        stream::enqueue(spStream, vEvents.back());
                    }

                    // Now wait for all the events.
                    for(auto && event : vEvents)
                    {
                        wait::wait(event);
                    }
                }
            };
        }
    }
}
