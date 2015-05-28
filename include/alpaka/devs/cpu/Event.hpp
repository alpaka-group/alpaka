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

#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu

#include <alpaka/traits/Event.hpp>          // StreamEnqueue, ...
#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor
#include <alpaka/traits/Dev.hpp>            // GetDev

#include <boost/uuid/uuid.hpp>              // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>   // boost::uuids::random_generator

#include <type_traits>                      // std::is_base
#include <mutex>                            // std::mutex
#include <condition_variable>               // std::condition_variable

namespace alpaka
{
    namespace devs
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device event implementation.
                //#############################################################################
                class EventCpuImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST EventCpuImpl(
                        DevCpu const & dev) :
                            m_Uuid(boost::uuids::random_generator()()),
                            m_Dev(dev),
                            m_Mutex(),
                            m_bIsReady(true),
                            m_bIsWaitedFor(false),
                            m_uiNumCanceledEnqueues(0)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST EventCpuImpl(EventCpuImpl const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST EventCpuImpl(EventCpuImpl &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(EventCpuImpl const &) -> EventCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(EventCpuImpl &&) -> EventCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~EventCpuImpl()
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    noexcept(false)
                    {
                        // If a event is enqueued to a stream and gets waited on but destructed before it is completed it is kept alive until completed.
                        // This can never happen.
                        if(m_bIsWaitedFor)
                        {
                            throw std::runtime_error("Assertion failure: Destruction of a referenced (waited for) EventCpuImpl!");
                        }
                    }
#else
                    noexcept(true) = default;
#endif
                public:
                    boost::uuids::uuid const m_Uuid;                        //!< The unique ID.
                    DevCpu const m_Dev;                                     //!< The device this event is bound to.

                    std::mutex mutable m_Mutex;                             //!< The mutex used to synchronize access to the event.

                    bool m_bIsReady;                                        //!< If the event is not waiting within a stream (not enqueued or already completed).
                    std::condition_variable mutable m_ConditionVariable;    //!< The condition signaling the event completion.

                    bool m_bIsWaitedFor;                                    //!< If a (one or multiple) streams wait for this event. The event can not be changed (deleted/re-enqueued) until completion.

                    std::size_t m_uiNumCanceledEnqueues;                    //!< The number of successive re-enqueues while it was already in the queue. Reset on completion.
                };
            }

            //#############################################################################
            //! The CPU device event.
            //#############################################################################
            class EventCpu final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventCpu(
                    DevCpu const & dev) :
                        m_spEventCpuImpl(std::make_shared<detail::EventCpuImpl>(dev))
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventCpu(EventCpu const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventCpu(EventCpu &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(EventCpu const &) -> EventCpu & = default;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(EventCpu &&) -> EventCpu & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(EventCpu const & rhs) const
                -> bool
                {
                    return (m_spEventCpuImpl->m_Uuid == rhs.m_spEventCpuImpl->m_Uuid);
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(EventCpu const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }

            public:
                std::shared_ptr<detail::EventCpuImpl> m_spEventCpuImpl;
            };
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The CPU device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cpu::EventCpu const & event)
                -> devs::cpu::DevCpu
                {
                    return event.m_spEventCpuImpl->m_Dev;
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CPU device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                devs::cpu::EventCpu>
            {
                using type = devs::cpu::EventCpu;
            };

            //#############################################################################
            //! The CPU device event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest<
                devs::cpu::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //! \return If the event is not waiting within a stream (not enqueued or already handled).
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto eventTest(
                    devs::cpu::EventCpu const & event)
                -> bool
                {
                    std::lock_guard<std::mutex> lk(event.m_spEventCpuImpl->m_Mutex);

                    return event.m_spEventCpuImpl->m_bIsReady;
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CPU device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately. 
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::EventCpu const & event)
                -> void
                {
                    alpaka::wait::wait(event.m_spEventCpuImpl);
                }
            };
            //#############################################################################
            //! The CPU device event implementation thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately. 
            //!
            //! NOTE: This method is for internal usage only.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                std::shared_ptr<devs::cpu::detail::EventCpuImpl>>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    std::shared_ptr<devs::cpu::detail::EventCpuImpl> const & spEventCpuImpl)
                -> void
                {
                    std::unique_lock<std::mutex> lk(spEventCpuImpl->m_Mutex);

                    if(!spEventCpuImpl->m_bIsReady)
                    {
                        spEventCpuImpl->m_bIsWaitedFor = true;
                        spEventCpuImpl->m_ConditionVariable.wait(
                            lk,
                            [spEventCpuImpl]{return spEventCpuImpl->m_bIsReady;});
                    }
                }
            };
        }
    }
}