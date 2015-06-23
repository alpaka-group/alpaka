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

#include <alpaka/dev/DevCpu.hpp>            // DevCpu
#include <alpaka/dev/Traits.hpp>            // GetDev
#include <alpaka/event/Traits.hpp>          // StreamEnqueue, ...
#include <alpaka/wait/Traits.hpp>           // CurrentThreadWaitFor
#include <alpaka/dev/Traits.hpp>            // GetDev

#include <boost/uuid/uuid.hpp>              // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>   // boost::uuids::random_generator

#include <type_traits>                      // std::is_base
#include <mutex>                            // std::mutex
#include <condition_variable>               // std::condition_variable

namespace alpaka
{
    namespace event
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device synchronous event implementation.
                //#############################################################################
                class EventCpuSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST EventCpuSyncImpl(
                        dev::DevCpu const & dev) :
                            m_Uuid(boost::uuids::random_generator()()),
                            m_Dev(dev)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST EventCpuSyncImpl(EventCpuSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST EventCpuSyncImpl(EventCpuSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(EventCpuSyncImpl const &) -> EventCpuSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(EventCpuSyncImpl &&) -> EventCpuSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~EventCpuSyncImpl() noexcept = default;

                public:
                    boost::uuids::uuid const m_Uuid;                        //!< The unique ID.
                    dev::DevCpu const m_Dev;                                //!< The device this event is bound to.
                };
            }
        }

        //#############################################################################
        //! The CPU device synchronous event.
        //#############################################################################
        class EventCpuSync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST EventCpuSync(
                dev::DevCpu const & dev) :
                    m_spEventCpuSyncImpl(std::make_shared<cpu::detail::EventCpuSyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST EventCpuSync(EventCpuSync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST EventCpuSync(EventCpuSync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(EventCpuSync const &) -> EventCpuSync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(EventCpuSync &&) -> EventCpuSync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator==(EventCpuSync const & rhs) const
            -> bool
            {
                return (m_spEventCpuSyncImpl->m_Uuid == rhs.m_spEventCpuSyncImpl->m_Uuid);
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator!=(EventCpuSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }

        public:
            std::shared_ptr<cpu::detail::EventCpuSyncImpl> m_spEventCpuSyncImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                event::EventCpuSync>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    event::EventCpuSync const & event)
                -> dev::DevCpu
                {
                    return event.m_spEventCpuSyncImpl->m_Dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                event::EventCpuSync>
            {
                using type = event::EventCpuSync;
            };

            //#############################################################################
            //! The CPU device event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest<
                event::EventCpuSync>
            {
                //-----------------------------------------------------------------------------
                //! \return If the event is not waiting within a stream (not enqueued or already handled).
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto eventTest(
                    event::EventCpuSync const & event)
                -> bool
                {
                    return true;
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                event::EventCpuSync>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    event::EventCpuSync const & event)
                -> void
                {
                    boost::ignore_unused(event);
                }
            };
        }
    }
}