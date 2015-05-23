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

#include <alpaka/devs/cuda/Dev.hpp>         // DevCuda

#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Dev.hpp>            // GetDev
#include <alpaka/traits/Event.hpp>
#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor

#include <alpaka/core/Cuda.hpp>             // ALPAKA_CUDA_RT_CHECK

#include <stdexcept>                        // std::runtime_error
#include <memory>                           // std::shared_ptr
#include <functional>                       // std::bind

namespace alpaka
{
    namespace devs
    {
        namespace cuda
        {
            template<
                typename TDim>
            class AccGpuCuda;

            //#############################################################################
            //! The CUDA device event.
            //#############################################################################
            class EventCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventCuda(
                    DevCuda const & dev,
                    bool bBusyWait = true) :
                        m_Dev(dev),
                        m_spCudaEvent(
                            new cudaEvent_t,
                            std::bind(&EventCuda::destroyEvent, std::placeholders::_1, std::ref(m_Dev)))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        m_Dev.m_iDevice));
                    // Create the event on the current device with the specified flags. Valid flags include:
                    // - cudaEventDefault: Default event creation flag.
                    // - cudaEventBlockingSync : Specifies that event should use blocking synchronization.A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                    // - cudaEventDisableTiming : Specifies that the created event does not need to record timing data.Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                    ALPAKA_CUDA_RT_CHECK(cudaEventCreateWithFlags(
                        m_spCudaEvent.get(),
                        (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventCuda(EventCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventCuda(EventCuda &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(EventCuda const &) -> EventCuda & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(EventCuda const & rhs) const
                -> bool
                {
                    return (*m_spCudaEvent.get()) == (*rhs.m_spCudaEvent.get());
                }
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(EventCuda const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST /*virtual*/ ~EventCuda() noexcept = default;

            private:
                //-----------------------------------------------------------------------------
                //! Destroys the shared event.
                //-----------------------------------------------------------------------------
                static auto destroyEvent(
                    cudaEvent_t * event,
                    DevCuda const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device. \TODO: Is setting the current device before cudaEventDestroy required?
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    // In case event has been recorded but has not yet been completed when cudaEventDestroy() is called, the function will return immediately
                    // and the resources associated with event will be released automatically once the device has completed event.
                    // -> No need to synchronize here.
                    ALPAKA_CUDA_RT_CHECK(cudaEventDestroy(*event));
                }

            public:
                std::shared_ptr<cudaEvent_t> m_spCudaEvent;
                DevCuda m_Dev;
            };
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The CUDA device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cuda::EventCuda>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cuda::EventCuda const & event)
                -> devs::cuda::DevCuda
                {
                    return event.m_Dev;
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CUDA device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                devs::cuda::EventCuda>
            {
                using type = devs::cuda::EventCuda;
            };

            //#############################################################################
            //! The CUDA device event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest<
                devs::cuda::EventCuda>
            {
                ALPAKA_FCT_HOST static auto eventTest(
                    devs::cuda::EventCuda const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for events on non current device.
                    auto const ret(
                        cudaEventQuery(
                            *event.m_spCudaEvent.get()));
                    if(ret == cudaSuccess)
                    {
                        return true;
                    }
                    else if(ret == cudaErrorNotReady)
                    {
                        return false;
                    }
                    else
                    {
                        throw std::runtime_error(("Unexpected return value '" + std::string(cudaGetErrorString(ret)) + "'from cudaEventQuery!"));
                    }
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CUDA device event thread wait trait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cuda::EventCuda>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cuda::EventCuda const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaEventSynchronize(
                        *event.m_spCudaEvent.get()));
                }
            };
        }
    }
}
