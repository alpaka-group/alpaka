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

#include <alpaka/traits/Stream.hpp>     // traits::StreamEnqueueEvent, ...
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor, WaiterWaitFor
#include <alpaka/traits/Acc.hpp>        // AccT
#include <alpaka/traits/Dev.hpp>        // GetDev

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::is_base

namespace alpaka
{
    namespace devs
    {
        namespace cpu
        {
            //#############################################################################
            //! The CPU device stream.
            //! NOTE: This stream is currently synchronous!
            //#############################################################################
            class StreamCpu
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCpu(
                    DevCpu & dev) :
                        m_Dev(dev)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCpu(StreamCpu const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCpu(StreamCpu &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(StreamCpu const &) -> StreamCpu & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(StreamCpu const & rhs) const
                -> bool
                {
                    return (m_Dev == rhs.m_Dev);
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(StreamCpu const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~StreamCpu() noexcept = default;

            public:
                DevCpu m_Dev;
            };

            class EventCpu;
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The CPU device stream device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cpu::StreamCpu>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cpu::StreamCpu const & stream)
                -> devs::cpu::DevCpu
                {
                    return stream.m_Dev;
                }
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CPU device stream stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                devs::cpu::StreamCpu>
            {
                using type = devs::cpu::StreamCpu;
            };

            //#############################################################################
            //! The CPU device stream test trait specialization.
            //#############################################################################
            template<>
            struct StreamTest<
                devs::cpu::StreamCpu>
            {
                ALPAKA_FCT_HOST static auto streamTest(
                    devs::cpu::StreamCpu const & stream)
                -> bool
                {
                    boost::ignore_unused(stream);
                    // Because cpu calls are not asynchronous, this call always returns true.
                    return true;
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CPU device stream thread wait trait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cpu::StreamCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::StreamCpu const & stream)
                -> void
                {
                    boost::ignore_unused(stream);
                    // Because cpu calls are not asynchronous, this call never has to wait.
                }
            };

            //#############################################################################
            //! The CPU device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                devs::cpu::StreamCpu,
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    devs::cpu::StreamCpu const & stream,
                    devs::cpu::EventCpu const & event)
                -> void
                {
                    boost::ignore_unused(stream);
                    boost::ignore_unused(event);
                    // Because cpu calls are not asynchronous, this call never has to let a stream wait.
                }
            };
        }
    }
}