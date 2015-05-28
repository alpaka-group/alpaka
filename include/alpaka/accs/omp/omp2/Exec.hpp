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
#include <alpaka/traits/Acc.hpp>                // AccType
#include <alpaka/traits/Exec.hpp>               // ExecType
#include <alpaka/traits/Event.hpp>              // EventType
#include <alpaka/traits/Dev.hpp>                // DevType
#include <alpaka/traits/Stream.hpp>             // StreamType

// Implementation details.
#include <alpaka/core/BasicWorkDiv.hpp>         // workdiv::BasicWorkDiv
#include <alpaka/core/NdLoop.hpp>               // NdLoop
#include <alpaka/accs/omp/omp2/Acc.hpp>         // AccCpuOmp2
#include <alpaka/accs/omp/Common.hpp>
#include <alpaka/devs/cpu/Dev.hpp>              // DevCpu
#include <alpaka/devs/cpu/Event.hpp>            // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>           // StreamCpu
#include <alpaka/traits/Kernel.hpp>             // BlockSharedExternMemSizeBytes

#include <cassert>                              // assert
#include <stdexcept>                            // std::runtime_error
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace accs
    {
        namespace omp
        {
            namespace omp2
            {
                namespace detail
                {
                    //#############################################################################
                    //! The CPU OpenMP2 accelerator executor implementation.
                    //#############################################################################
                    template<
                        typename TDim>
                    class ExecCpuOmp2Impl final
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecCpuOmp2Impl() = default;
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecCpuOmp2Impl(ExecCpuOmp2Impl const &) = default;
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecCpuOmp2Impl(ExecCpuOmp2Impl &&) = default;
                        //-----------------------------------------------------------------------------
                        //! Copy assignment operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2Impl const &) -> ExecCpuOmp2Impl & = default;
                        //-----------------------------------------------------------------------------
                        //! Move assignment operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2Impl &&) -> ExecCpuOmp2Impl & = default;
                        //-----------------------------------------------------------------------------
                        //! Destructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ~ExecCpuOmp2Impl() noexcept = default;

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

                            auto const vuiGridBlockExtents(
                                workdiv::getWorkDiv<Grid, Blocks>(workDiv));
                            auto const vuiBlockThreadExtents(
                                workdiv::getWorkDiv<Block, Threads>(workDiv));

                            auto const uiBlockSharedExternMemSizeBytes(
                                kernel::getBlockSharedExternMemSizeBytes<
                                    typename std::decay<TKernelFunctor>::type,
                                    AccCpuOmp2<TDim>>(
                                        vuiBlockThreadExtents,
                                        args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                                << std::endl;
#endif
                            AccCpuOmp2<TDim> acc(workDiv);

                            if(uiBlockSharedExternMemSizeBytes > 0)
                            {
                                acc.m_vuiExternalSharedMem.reset(
                                    new uint8_t[uiBlockSharedExternMemSizeBytes]);
                            }

                            // The number of threads in this block.
                            UInt const uiNumThreadsInBlock(vuiBlockThreadExtents.prod());
                            int const iNumThreadsInBlock(static_cast<int>(uiNumThreadsInBlock));

                            // Execute the blocks serially.
                            ndLoop(
                                vuiGridBlockExtents,
                                [&](Vec<TDim> const & vuiGridBlockIdx)
                                {
                                    acc.m_vuiGridBlockIdx = vuiGridBlockIdx;

                                    // Execute the threads in parallel.

                                    // Force the environment to use the given number of threads.
                                    ::omp_set_dynamic(0);

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
                                                throw std::runtime_error("The OpenMP2 runtime did not use the number of threads that had been required!");
                                            }
                                        }
#endif
#endif
                                        kernelFunctor(
                                            acc,
                                            args...);

                                        // Wait for all threads to finish before deleting the shared memory.
                                        acc.syncBlockThreads();
                                    }

                                    // After a block has been processed, the shared memory has to be deleted.
                                    acc.m_vvuiSharedMem.clear();
                                });

                            // After all blocks have been processed, the external shared memory has to be deleted.
                            acc.m_vuiExternalSharedMem.reset();
                        }
                    };

                    //#############################################################################
                    //! The CPU OpenMP2 accelerator executor.
                    //#############################################################################
                    template<
                        typename TDim>
                    class ExecCpuOmp2 final :
                        public workdiv::BasicWorkDiv<TDim>
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TWorkDiv>
                        ALPAKA_FCT_HOST ExecCpuOmp2(
                            TWorkDiv const & workDiv,
                            devs::cpu::StreamCpu & stream) :
                                workdiv::BasicWorkDiv<TDim>(workDiv),
                                m_Stream(stream)
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        }
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecCpuOmp2(ExecCpuOmp2 const &) = default;
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecCpuOmp2(ExecCpuOmp2 &&) = default;
                        //-----------------------------------------------------------------------------
                        //! Copy assignment operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2 const &) -> ExecCpuOmp2 & = default;
                        //-----------------------------------------------------------------------------
                        //! Move assignment operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator=(ExecCpuOmp2 &&) -> ExecCpuOmp2 & = default;
                        //-----------------------------------------------------------------------------
                        //! Destructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ~ExecCpuOmp2() noexcept = default;

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
                        
                            auto const & workDiv(*static_cast<workdiv::BasicWorkDiv<TDim> const *>(this));

                            m_Stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                                [workDiv, kernelFunctor, args...]()
                                {
                                    ExecCpuOmp2Impl<TDim> exec;
                                    exec(
                                        workDiv,
                                        kernelFunctor,
                                        args...);
                                });
                        }

                    public:
                        devs::cpu::StreamCpu m_Stream;
                    };
                }
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU OpenMP2 executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                using type = accs::omp::omp2::detail::AccCpuOmp2<TDim>;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU OpenMP2 executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP2 executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU OpenMP2 executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                using type = TDim;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CPU OpenMP2 executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct EventType<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                using type = devs::cpu::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU OpenMP2 executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                using type = accs::omp::omp2::detail::ExecCpuOmp2<TDim>;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CPU OpenMP2 executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct StreamType<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                using type = devs::cpu::StreamCpu;
            };
            //#############################################################################
            //! The CPU OpenMP2 executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetStream<
                accs::omp::omp2::detail::ExecCpuOmp2<TDim>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::omp::omp2::detail::ExecCpuOmp2<TDim> const & exec)
                -> devs::cpu::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
