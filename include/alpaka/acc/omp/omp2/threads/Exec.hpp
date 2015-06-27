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
#include <alpaka/acc/omp/omp2/threads/Acc.hpp>  // AccCpuOmp2Threads
#include <alpaka/dev/DevCpu.hpp>                // DevCpu
#include <alpaka/event/EventCpuAsync.hpp>       // EventCpuAsync
#include <alpaka/kernel/Traits.hpp>             // BlockSharedExternMemSizeBytes
#include <alpaka/stream/StreamCpuAsync.hpp>     // StreamCpuAsync
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/core/OpenMp.hpp>
#include <alpaka/core/NdLoop.hpp>               // NdLoop

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
                namespace threads
                {
                    namespace detail
                    {
                        //#############################################################################
                        //! The CPU OpenMP 2.0 thread accelerator executor implementation.
                        //#############################################################################
                        template<
                            typename TDim>
                        class ExecCpuOmp2ThreadsImpl final
                        {
                        public:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecCpuOmp2ThreadsImpl() = default;
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecCpuOmp2ThreadsImpl(ExecCpuOmp2ThreadsImpl const &) = default;
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecCpuOmp2ThreadsImpl(ExecCpuOmp2ThreadsImpl &&) = default;
                            //-----------------------------------------------------------------------------
                            //! Copy assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2ThreadsImpl const &) -> ExecCpuOmp2ThreadsImpl & = default;
                            //-----------------------------------------------------------------------------
                            //! Move assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2ThreadsImpl &&) -> ExecCpuOmp2ThreadsImpl & = default;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ~ExecCpuOmp2ThreadsImpl() = default;

                            //-----------------------------------------------------------------------------
                            //! Executes the kernel functor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv,
                                typename TKernelFunctor,
                                typename... TArgs>
                            ALPAKA_FCT_HOST auto operator()(
                                TWorkDiv const & workDiv,
                                TKernelFunctor const & kernelFunctor,
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
                                        typename std::decay<TKernelFunctor>::type,
                                        AccCpuOmp2Threads<TDim>>(
                                            vuiBlockThreadExtents,
                                            args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                std::cout << BOOST_CURRENT_FUNCTION
                                    << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                                    << std::endl;
#endif
                                AccCpuOmp2Threads<TDim> acc(workDiv);

                                if(uiBlockSharedExternMemSizeBytes > 0u)
                                {
                                    acc.m_vuiExternalSharedMem.reset(
                                        reinterpret_cast<uint8_t *>(
                                            boost::alignment::aligned_alloc(16u, uiBlockSharedExternMemSizeBytes)));
                                }

                                // The number of threads in this block.
                                Uint const uiNumThreadsInBlock(vuiBlockThreadExtents.prod());
                                int const iNumThreadsInBlock(static_cast<int>(uiNumThreadsInBlock));

                                // Force the environment to use the given number of threads.
                                int const iOmpIsDynamic(::omp_get_dynamic());
                                ::omp_set_dynamic(0);

                                // Execute the blocks serially.
                                ndLoop(
                                    vuiGridBlockExtents,
                                    [&](Vec<TDim> const & vuiGridBlockIdx)
                                    {
                                        acc.m_vuiGridBlockIdx = vuiGridBlockIdx;

                                        // Execute the threads in parallel.

                                        // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                        // So we have to spawn one OS thread per thread in a block.
                                        // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                        // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                        #pragma omp parallel num_threads(iNumThreadsInBlock)
                                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                            // GCC 5.1 fails with:
                                            // error: redeclaration of ‘const int& iNumThreadsInBlock’
                                            // if(iNumThreads != iNumThreadsInBloc
                                            //                ^
                                            // note: ‘const int& iNumThreadsInBlock’ previously declared here
                                            // #pragma omp parallel num_threads(iNumThread
                                            //         ^
#if (!BOOST_COMP_GNUC) || (BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(5, 0, 0))
                                            // The first thread does some checks in the first block executed.
                                            if((::omp_get_thread_num() == 0) && (acc.m_vuiGridBlockIdx.sum() == 0u))
                                            {
                                                int const iNumThreads(::omp_get_num_threads());
                                                std::cout << BOOST_CURRENT_FUNCTION << " omp_get_num_threads: " << iNumThreads << std::endl;
                                                if(iNumThreads != iNumThreadsInBlock)
                                                {
                                                    throw std::runtime_error("The OpenMP 2.0 runtime did not use the number of threads that had been required!");
                                                }
                                            }
#endif
#endif
                                            kernelFunctor(
                                                const_cast<AccCpuOmp2Threads<TDim> const &>(acc),
                                                args...);

                                            // Wait for all threads to finish before deleting the shared memory.
                                            acc.syncBlockThreads();
                                        }

                                        // After a block has been processed, the shared memory has to be deleted.
                                        block::shared::freeMem(acc);
                                    });

                                // After all blocks have been processed, the external shared memory has to be deleted.
                                acc.m_vuiExternalSharedMem.reset();

                                // Reset the dynamic thread number setting.
                                ::omp_set_dynamic(iOmpIsDynamic);
                            }
                        };
                    }
                }
            }
        }

        //#############################################################################
        //! The CPU OpenMP 2.0 thread accelerator executor.
        //#############################################################################
        template<
            typename TDim>
        class ExecCpuOmp2Threads final :
            public workdiv::WorkDivMembers<TDim>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST ExecCpuOmp2Threads(
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
            ALPAKA_FCT_HOST ExecCpuOmp2Threads(ExecCpuOmp2Threads const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ExecCpuOmp2Threads(ExecCpuOmp2Threads &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2Threads const &) -> ExecCpuOmp2Threads & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2Threads &&) -> ExecCpuOmp2Threads & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~ExecCpuOmp2Threads() = default;

            //-----------------------------------------------------------------------------
            //! Enqueues the kernel functor.
            //-----------------------------------------------------------------------------
            template<
                typename TKernelFunctor,
                typename... TArgs>
            ALPAKA_FCT_HOST auto operator()(
                TKernelFunctor const & kernelFunctor,
                TArgs const & ... args) const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const & workDiv(*static_cast<workdiv::WorkDivMembers<TDim> const *>(this));

                m_Stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                    [workDiv, kernelFunctor, args...]()
                    {
                        omp::omp2::threads::detail::ExecCpuOmp2ThreadsImpl<TDim> exec;
                        exec(
                            workDiv,
                            kernelFunctor,
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
            //! The CPU OpenMP 2.0 block thread executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                exec::ExecCpuOmp2Threads<TDim>>
            {
                using type = acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block thread executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                exec::ExecCpuOmp2Threads<TDim>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block thread executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                exec::ExecCpuOmp2Threads<TDim>>
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
            //! The CPU OpenMP 2.0 block thread executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                exec::ExecCpuOmp2Threads<TDim>>
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
            //! The CPU OpenMP 2.0 block thread executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct EventType<
                exec::ExecCpuOmp2Threads<TDim>>
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
            //! The CPU OpenMP 2.0 block thread executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                exec::ExecCpuOmp2Threads<TDim>>
            {
                using type = exec::ExecCpuOmp2Threads<TDim>;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block thread executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct StreamType<
                exec::ExecCpuOmp2Threads<TDim>>
            {
                using type = stream::StreamCpuAsync;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block thread executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetStream<
                exec::ExecCpuOmp2Threads<TDim>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    exec::ExecCpuOmp2Threads<TDim> const & exec)
                -> stream::StreamCpuAsync
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
