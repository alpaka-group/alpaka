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

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused
#include <boost/uuid/uuid.hpp>                  // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>       // boost::uuids::random_generator

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
                class StreamCpuSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCpuSyncImpl(
                        dev::DevCpu & dev) :
                            m_Uuid(boost::uuids::random_generator()()),
                            m_Dev(dev)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCpuSyncImpl(StreamCpuSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCpuSyncImpl(StreamCpuSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(StreamCpuSyncImpl const &) -> StreamCpuSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(StreamCpuSyncImpl &&) -> StreamCpuSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~StreamCpuSyncImpl() = default;// noexcept(false)
                    //{
                        //m_Dev.m_spDevCpuImpl->UnregisterStream(this);
                    //}
                public:
                    boost::uuids::uuid const m_Uuid;    //!< The unique ID.
                    dev::DevCpu const m_Dev;            //!< The device this stream is bound to.
                };
            }
        }

        //#############################################################################
        //! The CPU device stream.
        //#############################################################################
        class StreamCpuSync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST StreamCpuSync(
                dev::DevCpu & dev) :
                    m_spSyncStreamCpu(std::make_shared<cpu::detail::StreamCpuSyncImpl>(dev))
            {
                //dev.m_spDevCpuImpl->RegisterStream(m_spAsyncStreamCpu);
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST StreamCpuSync(StreamCpuSync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST StreamCpuSync(StreamCpuSync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(StreamCpuSync const &) -> StreamCpuSync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(StreamCpuSync &&) -> StreamCpuSync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator==(StreamCpuSync const & rhs) const
            -> bool
            {
                return (m_spSyncStreamCpu->m_Uuid == rhs.m_spSyncStreamCpu->m_Uuid);
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator!=(StreamCpuSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~StreamCpuSync() = default;

        public:
            std::shared_ptr<cpu::detail::StreamCpuSyncImpl> m_spSyncStreamCpu;
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
                stream::StreamCpuSync>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    stream::StreamCpuSync const & stream)
                -> dev::DevCpu
                {
                    return stream.m_spSyncStreamCpu->m_Dev;
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
                stream::StreamCpuSync>
            {
                using type = stream::StreamCpuSync;
            };

            //#############################################################################
            //! The CPU device stream test trait specialization.
            //#############################################################################
            template<>
            struct StreamTest<
                stream::StreamCpuSync>
            {
                ALPAKA_FCT_HOST static auto streamTest(
                    stream::StreamCpuSync const & stream)
                -> bool
                {
                    boost::ignore_unused(stream);
                    return true;
                }
            };
        }
    }
}

#include <alpaka/stream/cpu/sync/StreamEventTraits.hpp>
