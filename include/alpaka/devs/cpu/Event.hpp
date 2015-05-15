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

#include <alpaka/devs/cpu/Dev.hpp>      // DevCpu

#include <alpaka/traits/Event.hpp>      // StreamEnqueueEvent, ...
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor
#include <alpaka/traits/Dev.hpp>        // GetDev

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::is_base

namespace alpaka
{
    namespace devs
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The cpu device event.
                //#############################################################################
                class EventCpu
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST EventCpu(
                        DevCpu const & dev) :
                            m_Dev(dev)
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
                    //! Assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(EventCpu const &) -> EventCpu & = default;
                    //-----------------------------------------------------------------------------
                    //! Equality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator==(EventCpu const & rhs) const
                    -> bool
                    {
                        return (m_Dev == rhs.m_Dev);
                    }
                    //-----------------------------------------------------------------------------
                    //! Inequality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator!=(EventCpu const & rhs) const
                    -> bool
                    {
                        return !((*this) == rhs);
                    }
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST virtual ~EventCpu() noexcept = default;

                public:
                    DevCpu m_Dev;
                };

                class StreamCpu;
            }
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The cpu device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cpu::detail::EventCpu>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cpu::detail::EventCpu const & event)
                -> devs::cpu::detail::DevCpu
                {
                    return event.m_Dev;
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The cpu device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                devs::cpu::detail::EventCpu>
            {
                using type = devs::cpu::detail::EventCpu;
            };

            //#############################################################################
            //! The cpu device event enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueueEvent<
                devs::cpu::detail::EventCpu,
                devs::cpu::detail::StreamCpu>
            {
                ALPAKA_FCT_HOST static auto streamEnqueueEvent(
                    devs::cpu::detail::EventCpu const & event,
                    devs::cpu::detail::StreamCpu const & stream)
                -> void
                {
                    boost::ignore_unused(event);
                    boost::ignore_unused(stream);
                    // Because cpu calls are not asynchronous, this call never has to enqueue anything.
                }
            };

            //#############################################################################
            //! The cpu device event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest<
                devs::cpu::detail::EventCpu>
            {
                ALPAKA_FCT_HOST static auto eventTest(
                    devs::cpu::detail::EventCpu const & event)
                -> bool
                {
                    boost::ignore_unused(event);
                    // Because cpu calls are not asynchronous, this call always returns true.
                    return true;
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The cpu device event thread wait trait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cpu::detail::EventCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::detail::EventCpu const & event)
                -> void
                {
                    boost::ignore_unused(event);
                    // Because cpu calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}