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

#include <alpaka/devs/cpu/Stream.hpp>   // StreamCpu
#include <alpaka/devs/cpu/Event.hpp>    // EventCpu

#include <alpaka/traits/Wait.hpp>       // wait

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                 // std::cout
#endif

namespace alpaka
{
    namespace traits
    {
        namespace stream
        {
            //#############################################################################
            //! The CPU device enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueue<
                std::shared_ptr<devs::cpu::detail::StreamCpuImpl>,
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto streamEnqueue(
                    std::shared_ptr<devs::cpu::detail::StreamCpuImpl> & spStreamImpl,
                    devs::cpu::EventCpu & event)
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
                devs::cpu::StreamCpu,
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto streamEnqueue(
                    devs::cpu::StreamCpu & stream,
                    devs::cpu::EventCpu & event)
                -> void
                {
                    alpaka::stream::enqueue(stream.m_spAsyncStreamCpu, event);
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CPU device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                std::shared_ptr<devs::cpu::detail::StreamCpuImpl>,
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    std::shared_ptr<devs::cpu::detail::StreamCpuImpl> & spStream,
                    devs::cpu::EventCpu const & event)
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
                            alpaka::wait::wait(spEventCpuImpl);
                        });
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
                    devs::cpu::StreamCpu & stream,
                    devs::cpu::EventCpu const & event)
                -> void
                {
                    alpaka::wait::wait(stream.m_spAsyncStreamCpu, event);
                }
            };

            //#############################################################################
            //! The CPU device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                devs::cpu::DevCpu,
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    devs::cpu::DevCpu & dev,
                    devs::cpu::EventCpu const & event)
                -> void
                {
                    // Get all the streams on the device at the time of invocation.
                    // All streams added afterwards are ignored.
                    auto vspStreams(
                        dev.m_spDevCpuImpl->GetRegisteredStreams());

                    // Let all the streams wait for this event.
                    // \TODO: This should be done atomically for all streams. 
                    // Furthermore there should not even be a chance to enqueue something between getting the streams and adding our wait events!
                    for(auto && spStream : vspStreams)
                    {
                        alpaka::wait::wait(spStream, event);
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
                devs::cpu::StreamCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::StreamCpu const & stream)
                -> void
                {
                    devs::cpu::EventCpu event(
                        alpaka::dev::getDev(stream));
                    alpaka::stream::enqueue(
                        const_cast<devs::cpu::StreamCpu &>(stream),
                        event);
                    alpaka::wait::wait(
                        event);
                }
            };
        }
    }
}