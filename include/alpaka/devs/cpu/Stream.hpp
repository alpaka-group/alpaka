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

#include <alpaka/devs/cpu/Dev.hpp>              // DevCpu

#include <alpaka/traits/Stream.hpp>             // traits::StreamEnqueue, ...
#include <alpaka/traits/Wait.hpp>               // CurrentThreadWaitFor, WaiterWaitFor
#include <alpaka/traits/Acc.hpp>                // AccT
#include <alpaka/traits/Dev.hpp>                // GetDev

#include <alpaka/core/ConcurrentExecPool.hpp>   // ConcurrentExecPool

#include <boost/uuid/uuid.hpp>                  // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>       // boost::uuids::random_generator

#include <type_traits>                          // std::is_base
#include <thread>                               // std::thread
#include <mutex>                                // std::mutex

namespace alpaka
{
    namespace devs
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device stream implementation.
                //#############################################################################
                class StreamCpuImpl final
                {
                private:
                    //#############################################################################
                    //
                    //#############################################################################
                    using ThreadPool = alpaka::detail::ConcurrentExecPool<
                        std::thread,                // The concurrent execution type.
                        std::promise,               // The promise type.
                        void,                       // The type yielding the current concurrent execution.
                        std::mutex,                 // The mutex type to use. Only required if TbYield is true.
                        std::condition_variable,    // The condition variable type to use. Only required if TbYield is true.
                        false>;                     // If the threads should yield.

                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCpuImpl(
                        DevCpu & dev) :
                            m_Uuid(boost::uuids::random_generator()()),
                            m_Dev(dev),
                            m_workerThread(1u, 128u)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCpuImpl(StreamCpuImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCpuImpl(StreamCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(StreamCpuImpl const &) -> StreamCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(StreamCpuImpl &&) -> StreamCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~StreamCpuImpl() noexcept(false)
                    {
                        m_Dev.m_spDevCpuImpl->UnregisterStream(this);
                    }
                public:
                    boost::uuids::uuid const m_Uuid;    //!< The unique ID.
                    DevCpu const m_Dev;                 //!< The device this stream is bound to.

                    ThreadPool m_workerThread;
                };
            }

            //#############################################################################
            //! The CPU device stream.
            //#############################################################################
            class StreamCpu final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCpu(
                    DevCpu & dev) :
                        m_spAsyncStreamCpu(std::make_shared<detail::StreamCpuImpl>(dev))
                {
                    dev.m_spDevCpuImpl->RegisterStream(m_spAsyncStreamCpu);
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCpu(StreamCpu const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCpu(StreamCpu &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(StreamCpu const &) -> StreamCpu & = default;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(StreamCpu &&) -> StreamCpu & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(StreamCpu const & rhs) const
                -> bool
                {
                    return (m_spAsyncStreamCpu->m_Uuid == rhs.m_spAsyncStreamCpu->m_Uuid);
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
                ALPAKA_FCT_HOST ~StreamCpu() = default;

            public:
                std::shared_ptr<detail::StreamCpuImpl> m_spAsyncStreamCpu;
            };
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
                    return stream.m_spAsyncStreamCpu->m_Dev;
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
                    return stream.m_spAsyncStreamCpu->m_workerThread.isQueueEmpty();
                }
            };
        }
    }
}