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
        namespace event
        {
            //#############################################################################
            //! The CPU device event enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueueEvent<
                devs::cpu::EventCpu,
                devs::cpu::StreamCpu>
            {
                ALPAKA_FCT_HOST static auto streamEnqueueEvent(
                    devs::cpu::EventCpu & event,
                    devs::cpu::StreamCpu & stream)
                -> void
                {
                    // Setting the event state and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> lk(event.m_spEventCpuImpl->m_Mutex);

                    // This is a invariant: If the event is ready (not enqueued) there can not be anybody waiting for it.
                    assert(!(event.m_spEventCpuImpl->m_bIsReady && event.m_spEventCpuImpl->m_bIsWaitedFor));

                    // If it is enqueued ...
                    if(!event.m_spEventCpuImpl->m_bIsReady)
                    {
                        // ... and somebody is waiting for it, it can NOT be re-enqueued.
                        if(event.m_spEventCpuImpl->m_bIsWaitedFor)
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                            std::cout << BOOST_CURRENT_FUNCTION << "WARNING: The event to enqueue is already enqueued AND waited on. It can NOT be re-enqueued!" << std::endl;
#endif
                            return;
                        }
                        // ... and was enqueued before, increment the cancel counter.
                        else
                        {
                            ++event.m_spEventCpuImpl->m_uiNumCanceledEnqueues;
                        }
                    }
                    // If it is not enqueued, set its state to enqueued.
                    else
                    {
                        event.m_spEventCpuImpl->m_bIsReady = false;
                    }

                    // Enqueue a task that only resets the events flag if it is completed.
                    stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                        [&event]()
                        {
                            {
                                std::lock_guard<std::mutex> lk(event.m_spEventCpuImpl->m_Mutex);
                                // Nothing to do if it has been re-enqueued to a later position in the queue.
                                if(event.m_spEventCpuImpl->m_uiNumCanceledEnqueues > 0)
                                {
                                    --event.m_spEventCpuImpl->m_uiNumCanceledEnqueues;
                                    return;
                                }
                                else
                                {
                                    event.m_spEventCpuImpl->m_bIsWaitedFor = false;
                                    event.m_spEventCpuImpl->m_bIsReady = true;
                                }
                            }
                            event.m_spEventCpuImpl->m_ConditionVariable.notify_all();
                        });
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
                devs::cpu::StreamCpu,
                devs::cpu::EventCpu>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    devs::cpu::StreamCpu & stream,
                    devs::cpu::EventCpu const & event)
                -> void
                {
                    {
                        std::lock_guard<std::mutex> lk(event.m_spEventCpuImpl->m_Mutex);
                        event.m_spEventCpuImpl->m_bIsWaitedFor = true;
                    }

                    // Enqueue a task that waits for the given event.
                    stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                        [&event]()
                        {
                            alpaka::wait::wait(event);
                        });
                }
            };

            //#############################################################################
            //! The CPU device event wait trait specialization.
            //!
            //! Any future work submitted in any stream will wait for event to complete before beginning execution.
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
                    {
                        std::lock_guard<std::mutex> lk(event.m_spEventCpuImpl->m_Mutex);
                        event.m_spEventCpuImpl->m_bIsWaitedFor = true;
                    }
                    // \FIXME: implement alpaka::wait::wait(DevCpu, EventCpu)!
                    throw std::runtime_error("Error: Waiting for an EventCpu in all streams of a device 'alpaka::wait::wait(DevCpu, EventCpu)' is not implemented!");
                }
            };

            //#############################################################################
            //! The CPU device stream thread wait trait specialization.
            //!
            //! Halts execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cpu::StreamCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cpu::StreamCpu const & stream)
                -> void
                {
                    devs::cpu::EventCpu event(alpaka::dev::getDev(stream));
                    alpaka::event::enqueue(event, const_cast<devs::cpu::StreamCpu &>(stream));
                    alpaka::wait::wait(event);
                }
            };
        }
    }
}