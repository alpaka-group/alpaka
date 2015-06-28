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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/dev/Traits.hpp>                // DevType
#include <alpaka/event/Traits.hpp>              // EventType
#include <alpaka/exec/Traits.hpp>               // ExecType
#include <alpaka/stream/Traits.hpp>             // StreamType

// Implementation details.
#include <alpaka/acc/serial/Acc.hpp>            // AccCpuSerial
#include <alpaka/dev/DevCpu.hpp>                // DevCpu
#include <alpaka/event/EventCpuAsync.hpp>       // EventCpuAsync
#include <alpaka/kernel/Traits.hpp>             // BlockSharedExternMemSizeBytes
#include <alpaka/stream/StreamCpuAsync.hpp>     // StreamCpuAsync
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/core/NdLoop.hpp>               // NdLoop

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused
#include <boost/align.hpp>                      // boost::aligned_alloc

#include <cassert>                              // assert
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace exec
    {
        namespace serial
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU serial executor implementation.
                //#############################################################################
                template<
                    typename TDim>
                class ExecCpuSerialImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuSerialImpl() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuSerialImpl(ExecCpuSerialImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuSerialImpl(ExecCpuSerialImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuSerialImpl const &) -> ExecCpuSerialImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuSerialImpl &&) -> ExecCpuSerialImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~ExecCpuSerialImpl() = default;

                    //-----------------------------------------------------------------------------
                    //! Executes the kernel function object.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv,
                        typename TKernelFuncObj,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto operator()(
                        TWorkDiv const & workDiv,
                        TKernelFuncObj const & kernelFuncObj,
                        TArgs const & ... args) const
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TWorkDiv>::value == TDim::value,
                            "The work division and the executor have to of the same dimensionality!");

                        auto const vuiGridBlockExtents(
                            workdiv::getWorkDiv<Grid, Blocks>(workDiv));
                        auto const vuiBlockThreadExtents(
                            workdiv::getWorkDiv<Block, Threads>(workDiv));

                        auto const uiBlockSharedExternMemSizeBytes(
                            kernel::getBlockSharedExternMemSizeBytes<
                                typename std::decay<TKernelFuncObj>::type,
                                AccCpuSerial<TDim>>(
                                    vuiBlockThreadExtents,
                                    args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif
                        AccCpuSerial<TDim> acc(workDiv);

                        if(uiBlockSharedExternMemSizeBytes > 0u)
                        {
                            acc.m_vuiExternalSharedMem.reset(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(16u, uiBlockSharedExternMemSizeBytes)));
                        }

                        // There is only ever one thread in a block in the serial accelerator.
                        assert(vuiBlockThreadExtents.prod() == 1u);

                        // Execute the blocks serially.
                        ndLoop(
                            vuiGridBlockExtents,
                            [&](Vec<TDim> const & vuiBlockThreadIdx)
                            {
                                acc.m_vuiGridBlockIdx = vuiBlockThreadIdx;

                                kernelFuncObj(
                                    const_cast<AccCpuSerial<TDim> const &>(acc),
                                    args...);

                                // After a block has been processed, the shared memory has to be deleted.
                                block::shared::freeMem(acc);
                            });

                        // After all blocks have been processed, the external shared memory has to be deleted.
                        acc.m_vuiExternalSharedMem.reset();
                    }
                };
            }
        }

        //#############################################################################
        //! The CPU serial executor.
        //#############################################################################
        template<
            typename TDim>
        class ExecCpuSerial final :
            public workdiv::WorkDivMembers<TDim>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST ExecCpuSerial(
                TWorkDiv const & workDiv,
                stream::StreamCpuAsync & stream) :
                    workdiv::WorkDivMembers<TDim>(workDiv),
                    m_Stream(stream)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    dim::DimT<TWorkDiv>::value == TDim::value,
                    "The work division and the executor have to of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ExecCpuSerial(ExecCpuSerial const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ExecCpuSerial(ExecCpuSerial &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(ExecCpuSerial const &) -> ExecCpuSerial & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(ExecCpuSerial &&) -> ExecCpuSerial & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~ExecCpuSerial() = default;

            //-----------------------------------------------------------------------------
            //! Enqueues the kernel function object.
            //-----------------------------------------------------------------------------
            template<
                typename TKernelFuncObj,
                typename... TArgs>
            ALPAKA_FCT_HOST auto operator()(
                TKernelFuncObj const & kernelFuncObj,
                TArgs const & ... args) const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const & workDiv(*static_cast<workdiv::WorkDivMembers<TDim> const *>(this));

                m_Stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                    [workDiv, kernelFuncObj, args...]()
                    {
                        serial::detail::ExecCpuSerialImpl<TDim> exec;
                        exec(
                            workDiv,
                            kernelFuncObj,
                            args...);
                    });
            }

        public:
            stream::StreamCpuAsync m_Stream;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                exec::ExecCpuSerial<TDim>>
            {
                using type = acc::serial::detail::AccCpuSerial<TDim>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                exec::ExecCpuSerial<TDim>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU serial executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                exec::ExecCpuSerial<TDim>>
            {
                using type = dev::DevManCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                exec::ExecCpuSerial<TDim>>
            {
                using type = TDim;
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct EventType<
                exec::ExecCpuSerial<TDim>>
            {
                using type = event::EventCpuAsync;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                exec::ExecCpuSerial<TDim>>
            {
                using type = exec::ExecCpuSerial<TDim>;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct StreamType<
                exec::ExecCpuSerial<TDim>>
            {
                using type = stream::StreamCpuAsync;
            };
            //#############################################################################
            //! The CPU serial executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetStream<
                exec::ExecCpuSerial<TDim>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    exec::ExecCpuSerial<TDim> const & exec)
                -> stream::StreamCpuAsync
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
