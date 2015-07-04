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
#include <alpaka/size/Traits.hpp>               // size::SizeT
#include <alpaka/stream/Traits.hpp>             // StreamType

// Implementation details.
#include <alpaka/acc/omp/omp2/blocks/Acc.hpp>   // AccCpuOmp2Blocks
#include <alpaka/dev/DevCpu.hpp>                // DevCpu
#include <alpaka/event/EventCpuAsync.hpp>       // EventCpuAsync
#include <alpaka/kernel/Traits.hpp>             // BlockSharedExternMemSizeBytes
#include <alpaka/stream/StreamCpuAsync.hpp>     // StreamCpuAsync
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/core/OpenMp.hpp>
#include <alpaka/core/MapIdx.hpp>               // mapIdx

#include <boost/align.hpp>                      // boost::aligned_alloc

#include <cassert>                              // assert
#include <stdexcept>                            // std::runtime_error
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace exec
    {
        namespace omp
        {
            namespace omp2
            {
                namespace blocks
                {
                    namespace detail
                    {
                        //#############################################################################
                        //! The CPU OpenMP 2.0 block accelerator executor implementation.
                        //#############################################################################
                        template<
                            typename TDim,
                            typename TSize>
                        class ExecCpuOmp2BlocksImpl final
                        {
                        public:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_HOST ExecCpuOmp2BlocksImpl() = default;
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_HOST ExecCpuOmp2BlocksImpl(ExecCpuOmp2BlocksImpl const &) = default;
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_HOST ExecCpuOmp2BlocksImpl(ExecCpuOmp2BlocksImpl &&) = default;
                            //-----------------------------------------------------------------------------
                            //! Copy assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_HOST auto operator=(ExecCpuOmp2BlocksImpl const &) -> ExecCpuOmp2BlocksImpl & = default;
                            //-----------------------------------------------------------------------------
                            //! Move assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_HOST auto operator=(ExecCpuOmp2BlocksImpl &&) -> ExecCpuOmp2BlocksImpl & = default;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_HOST ~ExecCpuOmp2BlocksImpl() = default;

                            //-----------------------------------------------------------------------------
                            //! Executes the kernel function object.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv,
                                typename TKernelFnObj,
                                typename... TArgs>
                            ALPAKA_FN_HOST auto operator()(
                                TWorkDiv const & workDiv,
                                TKernelFnObj const & kernelFnObj,
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
                                        typename std::decay<TKernelFnObj>::type,
                                        AccCpuOmp2Blocks<TDim, TSize>>(
                                            vuiBlockThreadExtents,
                                            args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                std::cout << BOOST_CURRENT_FUNCTION
                                    << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                                    << std::endl;
#endif
                                // The number of blocks in the grid.
                                TSize const uiNumBlocksInGrid(vuiGridBlockExtents.prod());
                                // There is only ever one thread in a block in the OpenMP 2.0 block accelerator.
                                assert(vuiBlockThreadExtents.prod() == 1u);

                                // Force the environment to use the given number of threads.
                                int const iOmpIsDynamic(::omp_get_dynamic());
                                ::omp_set_dynamic(0);

                                // Execute the blocks in parallel.
                                // NOTE: Setting num_threads(number_of_cores) instead of the default thread number does not improve performance.
                                #pragma omp parallel
                                {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                    // The first thread does some debug logging.
                                    if(::omp_get_thread_num() == 0)
                                    {
                                        int const iNumThreads(::omp_get_num_threads());
                                        std::cout << BOOST_CURRENT_FUNCTION << " omp_get_num_threads: " << iNumThreads << std::endl;
                                    }
#endif
                                    AccCpuOmp2Blocks<TDim, TSize> acc(workDiv);

                                    if(uiBlockSharedExternMemSizeBytes > 0u)
                                    {
                                        acc.m_vuiExternalSharedMem.reset(
                                            reinterpret_cast<uint8_t *>(
                                                boost::alignment::aligned_alloc(16u, uiBlockSharedExternMemSizeBytes)));
                                    }

                                    // NOTE: schedule(static) does not improve performance.
#if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop header.
                                    std::intmax_t i;
                                    #pragma omp for nowait
                                    for(i = 0; i < uiNumBlocksInGrid; ++i)
#else
                                    #pragma omp for nowait
                                    for(TSize i = 0; i < uiNumBlocksInGrid; ++i)
#endif
                                    {
                                        acc.m_vuiGridBlockIdx =
                                            mapIdx<TDim::value>(
#if _OPENMP < 200805
                                                Vec1<TSize>(static_cast<TSize>(i)),
#else
                                                Vec1<TSize>(i),
#endif
                                                vuiGridBlockExtents);

                                        kernelFnObj(
                                            const_cast<AccCpuOmp2Blocks<TDim, TSize> const &>(acc),
                                            args...);

                                        // After a block has been processed, the shared memory has to be deleted.
                                        block::shared::freeMem(acc);
                                    }

                                    // After all blocks have been processed, the external shared memory has to be deleted.
                                    acc.m_vuiExternalSharedMem.reset();
                                }

                                // Reset the dynamic thread number setting.
                                ::omp_set_dynamic(iOmpIsDynamic);
                            }
                        };
                    }
                }
            }
        }

        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class ExecCpuOmp2Blocks final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuOmp2Blocks(
                TWorkDiv const & workDiv,
                stream::StreamCpuAsync & stream) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
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
            ALPAKA_FN_HOST ExecCpuOmp2Blocks(ExecCpuOmp2Blocks const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuOmp2Blocks(ExecCpuOmp2Blocks &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOmp2Blocks const &) -> ExecCpuOmp2Blocks & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOmp2Blocks &&) -> ExecCpuOmp2Blocks & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ExecCpuOmp2Blocks() = default;

            //-----------------------------------------------------------------------------
            //! Enqueues the kernel function object.
            //-----------------------------------------------------------------------------
            template<
                typename TKernelFnObj,
                typename... TArgs>
            ALPAKA_FN_HOST auto operator()(
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const & workDiv(*static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this));

                m_Stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                    [workDiv, kernelFnObj, args...]()
                    {
                        omp::omp2::blocks::detail::ExecCpuOmp2BlocksImpl<TDim, TSize> exec;
                        exec(
                            workDiv,
                            kernelFnObj,
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
            //! The CPU OpenMP 2.0 grid block executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
            {
                using type = acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 grid block executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 grid block executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct EventType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 grid block executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
            {
                using type = exec::ExecCpuOmp2Blocks<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct StreamType<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
            {
                using type = stream::StreamCpuAsync;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetStream<
                exec::ExecCpuOmp2Blocks<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getStream(
                    exec::ExecCpuOmp2Blocks<TDim, TSize> const & exec)
                -> stream::StreamCpuAsync
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
