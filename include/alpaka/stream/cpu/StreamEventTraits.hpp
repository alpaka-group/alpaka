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

#include <alpaka/event/EventCpuAsync.hpp>   // EventCpuAsync
#include <alpaka/stream/StreamCpuAsync.hpp> // StreamCpuAsync

#include <alpaka/wait/Traits.hpp>           // wait

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                     // std::cout
#endif

namespace alpaka
{
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueue<
                std::shared_ptr<stream::cpu::detail::StreamCpuImpl>,
                event::EventCpuAsync>
            {
                ALPAKA_FCT_HOST static auto streamEnqueue(
                    std::shared_ptr<stream::cpu::detail::StreamCpuImpl> & spStreamImpl,
                    event::EventCpuAsync & event)
                -> void
                {
                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventCpuImpl(event.m_spEventCpuImpl);

                    // Setting the event state and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> lk(spEventCpuImpl->m_Mutex);

                    // This is a invariant: If the event is ready (not enqueued) there can not be anybody waiting for it.
                    assert(!(spEventCpuImpl->m_bIsReady && spEventCpuImpl->m_bIsWaitedFor));

                    // If it is enqueued ...
                    if(!spEventCpuImpl->m_bIsReady)
                    {
                        // ... and somebody is waiting for it, it can NOT be re-enqueued.
                        if(spEventCpuImpl->m_bIsWaitedFor)
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                            std::cout << BOOST_CURRENT_FUNCTION << "WARNING: The event to enqueue is already enqueued AND waited on. It can NOT be re-enqueued!" << std::endl;
#endif
                            return;
                        }
                        // ... and was enqueued before, increment the cancel counter.
                        else
                        {
                            ++spEventCpuImpl->m_uiNumCanceledEnqueues;
                        }
                    }
                    // If it is not enqueued, set its state to enqueued.
                    else
                    {
                        spEventCpuImpl->m_bIsReady = false;
                    }

                    // Enqueue a task that only resets the events flag if it is completed.
                    spStreamImpl->m_workerThread.enqueueTask(
                        [spEventCpuImpl]()
                        {
                            {
                                std::lock_guard<std::mutex> lk(spEventCpuImpl->m_Mutex);
                                // Nothing to do if it has been re-enqueued to a later position in the queue.
                                if(spEventCpuImpl->m_uiNumCanceledEnqueues > 0)
                                {
                                    --spEventCpuImpl->m_uiNumCanceledEnqueues;
                                    return;
                                }
                                else
                                {
                                    spEventCpuImpl->m_bIsWaitedFor = false;
                                    spEventCpuImpl->m_bIsReady = true;
                                }
                            }
                            spEventCpuImpl->m_ConditionVariable.notify_all();
                        });
                }
            };
            //#############################################################################
            //! The CPU device enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueue<
                stream::StreamCpuAsync,
                event::EventCpuAsync>
            {
                ALPAKA_FCT_HOST static auto streamEnqueue(
                    stream::StreamCpuAsync & stream,
                    event::EventCpuAsync & event)
                -> void
                {
                    stream::enqueue(stream.m_spAsyncStreamCpu, event);
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
                std::shared_ptr<stream::cpu::detail::StreamCpuImpl>,
                event::EventCpuAsync>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    std::shared_ptr<stream::cpu::detail::StreamCpuImpl> & spStream,
                    event::EventCpuAsync const & event)
                -> void
                {
                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventCpuImpl(event.m_spEventCpuImpl);

                    {
                        std::lock_guard<std::mutex> lk(spEventCpuImpl->m_Mutex);
                        spEventCpuImpl->m_bIsWaitedFor = true;
                    }

                    // Enqueue a task that waits for the given event.
                    spStream->m_workerThread.enqueueTask(
                        [spEventCpuImpl]()
                        {
                            wait::wait(spEventCpuImpl);
                        });
                }
            };
            //#############################################################################
            //! The CPU device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamCpuAsync,
                event::EventCpuAsync>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    stream::StreamCpuAsync & stream,
                    event::EventCpuAsync const & event)
                -> void
                {
                    wait::wait(stream.m_spAsyncStreamCpu, event);
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
                event::EventCpuAsync>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    dev::DevCpu & dev,
                    event::EventCpuAsync const & event)
                -> void
                {
                    // Get all the streams on the device at the time of invocation.
                    // All streams added afterwards are ignored.
                    auto vspStreams(
                        dev.m_spDevCpuImpl->GetAllStreams());

                    // Let all the streams wait for this event.
                    // \TODO: This should be done atomically for all streams. 
                    // Furthermore there should not even be a chance to enqueue something between getting the streams and adding our wait events!
                    for(auto && spStream : vspStreams)
                    {
                        wait::wait(spStream, event);
                    }
                }
            };

            //#############################################################################
            //! The CPU device stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCpuAsync>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    stream::StreamCpuAsync const & stream)
                -> void
                {
                    event::EventCpuAsync event(
                        dev::getDev(stream));
                    stream::enqueue(
                        const_cast<stream::StreamCpuAsync &>(stream),
                        event);
                    wait::wait(
                        event);
                }
            };
        }
    }
}