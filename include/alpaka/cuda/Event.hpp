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

#include <alpaka/traits/Event.hpp>
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor
#include <alpaka/traits/Acc.hpp>        // AccType

#include <alpaka/cuda/AccCudaFwd.hpp>   // AccCuda
#include <alpaka/cuda/Common.hpp>       // ALPAKA_CUDA_CHECK

#include <stdexcept>                    // std::runtime_error
#include <memory>                       // std::shared_ptr

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator event.
            //#############################################################################
            class EventCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventCuda(bool bBusyWait = true) :
                    m_spCudaEvent(new cudaEvent_t, &EventCuda::destroyEvent)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Creates an event object with the specified flags. Valid flags include:
                    //  cudaEventDefault: Default event creation flag.
                    //  cudaEventBlockingSync : Specifies that event should use blocking synchronization.A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                    //  cudaEventDisableTiming : Specifies that the created event does not need to record timing data.Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                    ALPAKA_CUDA_CHECK(cudaEventCreateWithFlags(
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
                ALPAKA_FCT_HOST EventCuda & operator=(EventCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(EventCuda const & rhs) const
                {
                    return (*m_spCudaEvent.get()) == (*rhs.m_spCudaEvent.get());
                }
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(EventCuda const & rhs) const
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
                static void destroyEvent(
                    cudaEvent_t * event)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // In case event has been recorded but has not yet been completed when cudaEventDestroy() is called, the function will return immediately 
                    // and the resources associated with event will be released automatically once the device has completed event.
                    // -> No need to synchronize here.
                    ALPAKA_CUDA_CHECK(cudaEventDestroy(*event));
                }

            public:
                std::shared_ptr<cudaEvent_t> m_spCudaEvent;
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CUDA accelerator event accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                cuda::detail::EventCuda>
            {
                using type = AccCuda;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CUDA accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                AccCuda>
            {
                using type = cuda::detail::EventCuda;
            };

            //#############################################################################
            //! The CUDA accelerator event enqueue trait specialization.
            //#############################################################################
            template<>
            struct DefaultStreamEnqueueEvent<
                cuda::detail::EventCuda>
            {
                ALPAKA_FCT_HOST static void defaultStreamEnqueueEvent(
                    cuda::detail::EventCuda const & event)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_CHECK(cudaEventRecord(
                        *event.m_spCudaEvent.get(),
                        nullptr));
                }
            };

            //#############################################################################
            //! The CUDA accelerator event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest<
                cuda::detail::EventCuda>
            {
                ALPAKA_FCT_HOST static bool eventTest(
                    cuda::detail::EventCuda const & event)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

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
            //! The CUDA accelerator event thread wait trait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                cuda::detail::EventCuda>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    cuda::detail::EventCuda const & event)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_CHECK(cudaEventSynchronize(*event.m_spCudaEvent.get()));
                }
            };
        }
    }
}
