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

#include <alpaka/dev/Traits.hpp>                // GetDev
#include <alpaka/dev/DevCpu.hpp>                // DevCpu
#include <alpaka/stream/Traits.hpp>             // stream::StreamEnqueue, ...
#include <alpaka/wait/Traits.hpp>               // CurrentThreadWaitFor, WaiterWaitFor

#include <alpaka/core/ConcurrentExecPool.hpp>   // ConcurrentExecPool

#include <boost/uuid/uuid.hpp>                  // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>       // boost::uuids::random_generator

#include <type_traits>                          // std::is_base
#include <thread>                               // std::thread
#include <mutex>                                // std::mutex

namespace alpaka
{
    namespace stream
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
                        dev::DevCpu & dev) :
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
                    dev::DevCpu const m_Dev;            //!< The device this stream is bound to.

                    ThreadPool m_workerThread;
                };
            }
        }

        //#############################################################################
        //! The CPU device stream.
        //#############################################################################
        class StreamCpuAsync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST StreamCpuAsync(
                dev::DevCpu & dev) :
                    m_spAsyncStreamCpu(std::make_shared<cpu::detail::StreamCpuImpl>(dev))
            {
                dev.m_spDevCpuImpl->RegisterStream(m_spAsyncStreamCpu);
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST StreamCpuAsync(StreamCpuAsync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST StreamCpuAsync(StreamCpuAsync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(StreamCpuAsync const &) -> StreamCpuAsync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(StreamCpuAsync &&) -> StreamCpuAsync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator==(StreamCpuAsync const & rhs) const
            -> bool
            {
                return (m_spAsyncStreamCpu->m_Uuid == rhs.m_spAsyncStreamCpu->m_Uuid);
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator!=(StreamCpuAsync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~StreamCpuAsync() = default;

        public:
            std::shared_ptr<cpu::detail::StreamCpuImpl> m_spAsyncStreamCpu;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device stream device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                stream::StreamCpuAsync>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    stream::StreamCpuAsync const & stream)
                -> dev::DevCpu
                {
                    return stream.m_spAsyncStreamCpu->m_Dev;
                }
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device stream stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                stream::StreamCpuAsync>
            {
                using type = stream::StreamCpuAsync;
            };

            //#############################################################################
            //! The CPU device stream test trait specialization.
            //#############################################################################
            template<>
            struct StreamTest<
                stream::StreamCpuAsync>
            {
                ALPAKA_FCT_HOST static auto streamTest(
                    stream::StreamCpuAsync const & stream)
                -> bool
                {
                    return stream.m_spAsyncStreamCpu->m_workerThread.isQueueEmpty();
                }
            };
        }
    }
}

#include <alpaka/stream/cpu/StreamEventTraits.hpp>
