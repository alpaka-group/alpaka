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
#include <alpaka/acc/omp/omp4/cpu/Acc.hpp>      // AccCpuOmp4
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
            namespace omp4
            {
                namespace cpu
                {
                    namespace detail
                    {
                        //#############################################################################
                        //! The CPU OpenMP 4.0 accelerator executor implementation.
                        //#############################################################################
                        template<
                            typename TDim,
                            typename TSize>
                        class ExecCpuOmp4Impl final
                        {
                        public:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecCpuOmp4Impl() = default;
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecCpuOmp4Impl(ExecCpuOmp4Impl const & other) = default;
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecCpuOmp4Impl(ExecCpuOmp4Impl && other) = default;
                            //-----------------------------------------------------------------------------
                            //! Copy assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp4Impl const &) -> ExecCpuOmp4Impl & = default;
                            //-----------------------------------------------------------------------------
                            //! Move assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp4Impl &&) -> ExecCpuOmp4Impl & = default;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ~ExecCpuOmp4Impl() = default;

                            //-----------------------------------------------------------------------------
                            //! Executes the kernel function object.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv,
                                typename TKernelFctObj,
                                typename... TArgs>
                            ALPAKA_FCT_HOST auto operator()(
                                TWorkDiv const & workDiv,
                                TKernelFctObj const & kernelFctObj,
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
                                        typename std::decay<TKernelFctObj>::type,
                                        AccCpuOmp4<TDim, TSize>>(
                                            vuiBlockThreadExtents,
                                            args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                std::cout << BOOST_CURRENT_FUNCTION
                                    << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                                    << std::endl;
#endif
                                // The number of blocks in the grid.
                                TSize const uiNumBlocksInGrid(vuiGridBlockExtents.prod());
                                int const iNumBlocksInGrid(static_cast<int>(uiNumBlocksInGrid));
                                // The number of threads in a block.
                                TSize const uiNumThreadsInBlock(vuiBlockThreadExtents.prod());
                                int const iNumThreadsInBlock(static_cast<int>(uiNumThreadsInBlock));

                                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                                #pragma omp target if(0)
                                {
                                    #pragma omp teams/* num_teams(iNumBlocksInGrid) thread_limit(iNumThreadsInBlock)*/
                                    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                        // The first team does some checks ...
                                        if((::omp_get_team_num() == 0))
                                        {
                                            int const iNumTeams(::omp_get_num_teams());
                                            // NOTE: No std::cout in omp target!
                                            printf("%s omp_get_num_teams: %d\n", BOOST_CURRENT_FUNCTION, iNumTeams);
                                            if(iNumTeams <= 0)    // NOTE: No throw inside target region
                                            {
                                                throw std::runtime_error("The CPU OpenMP4 runtime did not use a valid number of teams!");
                                            }
                                        }
#endif
                                        AccCpuOmp4<TDim, TSize> acc(workDiv);

                                        if(uiBlockSharedExternMemSizeBytes > 0u)
                                        {
                                            acc.m_vuiExternalSharedMem.reset(
                                                reinterpret_cast<uint8_t *>(
                                                    boost::alignment::aligned_alloc(16u, uiBlockSharedExternMemSizeBytes)));
                                        }

                                        #pragma omp distribute
                                        for(TSize b = 0u; b<uiNumBlocksInGrid; ++b)
                                        {
                                            Vec1<TSize> const v1iIdxGridBlock(b);
                                            // When this is not repeated here:
                                            // error: ‘vuiGridBlockExtents’ referenced in target region does not have a mappable type
                                            auto const vuiGridBlockExtents2(
                                                workdiv::getWorkDiv<Grid, Blocks>(workDiv));
                                            acc.m_vuiGridBlockIdx = mapIdx<TDim::value>(
                                                v1iIdxGridBlock,
                                                vuiGridBlockExtents2);

                                            // Execute the threads in parallel.

                                            // Force the environment to use the given number of threads.
                                            int const iOmpIsDynamic(::omp_get_dynamic());
                                            ::omp_set_dynamic(0);

                                            // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                            // So we have to spawn one OS thread per thread in a block.
                                            // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                            // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                            #pragma omp parallel num_threads(iNumThreadsInBlock)
                                            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                                // The first thread does some checks in the first block executed.
                                                if((::omp_get_thread_num() == 0) && (b == 0))
                                                {
                                                    int const iNumThreads(::omp_get_num_threads());
                                                    // NOTE: No std::cout in omp target!
                                                    printf("%s omp_get_num_threads: %d\n", BOOST_CURRENT_FUNCTION, iNumThreads);
                                                    if(iNumThreads != iNumThreadsInBlock)
                                                    {
                                                        throw std::runtime_error("The CPU OpenMP4 runtime did not use the number of threads that had been required!");
                                                    }
                                                }
#endif
                                                kernelFctObj(
                                                    const_cast<AccCpuOmp4<TDim, TSize> const &>(acc),
                                                    args...);

                                                // Wait for all threads to finish before deleting the shared memory.
                                                acc.syncBlockThreads();
                                            }

                                            // Reset the dynamic thread number setting.
                                            ::omp_set_dynamic(iOmpIsDynamic);

                                            // After a block has been processed, the shared memory has to be deleted.
                                            block::shared::freeMem(acc);
                                        }
                                        // After all blocks have been processed, the external shared memory has to be deleted.
                                        acc.m_vuiExternalSharedMem.reset();
                                    }
                                }
                            }
                        };
                    }
                }
            }
        }

        //#############################################################################
        //! The CPU OpenMP 4.0 accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class ExecCpuOmp4 final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST ExecCpuOmp4(
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
            ALPAKA_FCT_HOST ExecCpuOmp4(ExecCpuOmp4 const & other) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ExecCpuOmp4(ExecCpuOmp4 && other) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp4 const &) -> ExecCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp4 &&) -> ExecCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~ExecCpuOmp4() = default;

            //-----------------------------------------------------------------------------
            //! Enqueues the kernel function object.
            //-----------------------------------------------------------------------------
            template<
                typename TKernelFctObj,
                typename... TArgs>
            ALPAKA_FCT_HOST auto operator()(
                TKernelFctObj const & kernelFctObj,
                TArgs const & ... args) const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const & workDiv(*static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this));

                m_Stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                    [workDiv, kernelFctObj, args...]()
                    {
                        omp::omp4::cpu::detail::ExecCpuOmp4Impl<TDim, TSize> exec;
                        exec(
                            workDiv,
                            kernelFctObj,
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
            //! The CPU OpenMP4 executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                exec::ExecCpuOmp4<TDim, TSize>>
            {
                using type = acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP4 executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                exec::ExecCpuOmp4<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP4 executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                exec::ExecCpuOmp4<TDim, TSize>>
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
            //! The CPU OpenMP4 executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                exec::ExecCpuOmp4<TDim, TSize>>
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
            //! The CPU OpenMP4 executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct EventType<
                exec::ExecCpuOmp4<TDim, TSize>>
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
            //! The CPU OpenMP4 executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                exec::ExecCpuOmp4<TDim, TSize>>
            {
                using type = exec::ExecCpuOmp4<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                exec::ExecCpuOmp4<TDim, TSize>>
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
            //! The CPU OpenMP4 executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct StreamType<
                exec::ExecCpuOmp4<TDim, TSize>>
            {
                using type = stream::StreamCpuAsync;
            };
            //#############################################################################
            //! The CPU OpenMP4 executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetStream<
                exec::ExecCpuOmp4<TDim, TSize>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    exec::ExecCpuOmp4<TDim, TSize> const & exec)
                -> stream::StreamCpuAsync
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
