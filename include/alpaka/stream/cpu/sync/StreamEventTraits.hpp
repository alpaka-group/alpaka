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

#include <alpaka/event/EventCpuSync.hpp>    // EventCpuSync
#include <alpaka/stream/StreamCpuSync.hpp>  // StreamCpuSync

#include <alpaka/wait/Traits.hpp>           // wait

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

namespace alpaka
{
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device stream event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                stream::StreamCpuSync>
            {
                using type = event::EventCpuSync;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueue<
                stream::StreamCpuSync,
                event::EventCpuSync>
            {
                ALPAKA_FCT_HOST static auto streamEnqueue(
                    stream::StreamCpuSync & stream,
                    event::EventCpuSync & event)
                -> void
                {
                    boost::ignore_unused(stream);
                    boost::ignore_unused(event);
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamCpuSync,
                event::EventCpuSync>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    stream::StreamCpuSync & stream,
                    event::EventCpuSync const & event)
                -> void
                {
                    boost::ignore_unused(stream);
                    boost::ignore_unused(event);
                }
            };

            //#############################################################################
            //! The CPU device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                dev::DevCpu,
                event::EventCpuSync>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    dev::DevCpu & dev,
                    event::EventCpuSync const & event)
                -> void
                {
                    boost::ignore_unused(dev);
                    boost::ignore_unused(event);
                }
            };

            //#############################################################################
            //! The CPU device stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCpuSync>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    stream::StreamCpuSync const & stream)
                -> void
                {
                    boost::ignore_unused(stream);
                }
            };
        }
    }
}