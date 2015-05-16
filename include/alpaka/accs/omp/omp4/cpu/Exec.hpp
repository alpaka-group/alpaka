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
#include <alpaka/accs/omp/omp4/cpu/Acc.hpp>     // AccOmp4Cpu
#include <alpaka/accs/omp/Common.hpp>
#include <alpaka/devs/cpu/Dev.hpp>              // DevCpu
#include <alpaka/devs/cpu/Event.hpp>            // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>           // StreamCpu
#include <alpaka/traits/Kernel.hpp>             // BlockSharedExternMemSizeBytes

#include <cassert>                              // assert
#include <stdexcept>                            // std::runtime_error
#include <utility>                              // std::forward
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace accs
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
                        //! The OpenMP4 CPU accelerator executor.
                        //#############################################################################
                        class ExecOmp4Cpu :
                            public alpaka::workdiv::BasicWorkDiv
                        {
                        public:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FCT_HOST ExecOmp4Cpu(
                                TWorkDiv const & workDiv,
                                devs::cpu::detail::StreamCpu & stream) :
                                    alpaka::workdiv::BasicWorkDiv(workDiv),
                                    m_Stream(stream)
                            {
                                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                            }
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecOmp4Cpu(ExecOmp4Cpu const & other) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST ExecOmp4Cpu(ExecOmp4Cpu && other) = default;
#endif
                            //-----------------------------------------------------------------------------
                            //! Copy assignment.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_HOST auto operator=(ExecOmp4Cpu const &) -> ExecOmp4Cpu & = delete;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                            ALPAKA_FCT_HOST virtual ~ExecOmp4Cpu() = default;
#else
                            ALPAKA_FCT_HOST virtual ~ExecOmp4Cpu() noexcept = default;
#endif
                            //-----------------------------------------------------------------------------
                            //! Executes the kernel functor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TKernelFunctor,
                                typename... TArgs>
                            ALPAKA_FCT_HOST auto operator()(
                                TKernelFunctor && kernelFunctor,
                                TArgs && ... args) const
                            -> void
                            {
                                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                                Vec3<> const v3uiGridBlockExtents(
                                    workdiv::getWorkDiv<Grid, Blocks, dim::Dim3>(
                                        *static_cast<workdiv::BasicWorkDiv const *>(this)));
                                Vec3<> const v3uiBlockThreadExtents(
                                    workdiv::getWorkDiv<Block, Threads, dim::Dim3>(
                                        *static_cast<workdiv::BasicWorkDiv const *>(this)));

                                auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccOmp4Cpu>(
                                    v3uiBlockThreadExtents,
                                    std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                std::cout << BOOST_CURRENT_FUNCTION
                                    << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                                    << std::endl;
#endif
                                // The number of blocks in the grid.
                                auto const uiNumBlocksInGrid(
                                    workdiv::getWorkDiv<Grid, Blocks, dim::Dim1>(
                                        *static_cast<workdiv::BasicWorkDiv const *>(this))[0]);
                                // The number of threads in a block.
                                auto const uiNumThreadsInBlock(
                                    workdiv::getWorkDiv<Block, Threads, dim::Dim1>(
                                        *static_cast<workdiv::BasicWorkDiv const *>(this))[0]);

                                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                                #pragma omp target if(0)
                                {
                                    #pragma omp teams num_teams(static_cast<int>(uiNumBlocksInGrid)) thread_limit(static_cast<int>(uiNumThreadsInBlock))
                                    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                        if(::omp_get_team_num() == 0)
                                        {
                                            auto const uiNumTeams(static_cast<decltype(uiNumBlocksInGrid)>(::omp_get_num_teams()));
                                            printf("%s omp_get_num_teams: %d\n", BOOST_CURRENT_FUNCTION, uiNumTeams);
                                            // This can happen and is no problem for us.
                                            /*if(uiNumTeams != uiNumBlocksInGrid)
                                            {
                                                throw std::runtime_error("The OpenMP4 CPU runtime did not use the number of teams that had been required!");
                                            }*/
                                        }
#endif
                                        AccOmp4Cpu acc(*static_cast<workdiv::BasicWorkDiv const *>(this));

                                        if(uiBlockSharedExternMemSizeBytes > 0)
                                        {
                                            acc.m_vuiExternalSharedMem.reset(
                                                new uint8_t[uiBlockSharedExternMemSizeBytes]);
                                        }

                                        #pragma omp distribute
                                        for(UInt b = 0u; b<uiNumBlocksInGrid; ++b)
                                        {
                                            Vec1<> const v1iIdxGridBlock(b);
                                            auto const v3uiGridBlockExtents(workdiv::getWorkDiv<Grid, Blocks, dim::Dim3>(*static_cast<workdiv::BasicWorkDiv const *>(this)));
                                            auto const v2uiGridBlockExtents(subVecEnd<dim::Dim2>(v3uiGridBlockExtents));
                                            acc.m_v3uiGridBlockIdx = mapIdx<3>(
                                                v1iIdxGridBlock,
                                                v2uiGridBlockExtents);

                                            // Execute the threads in parallel.

                                            // Force the environment to use the given number of threads.
                                            ::omp_set_dynamic(0);

                                            // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                            // So we have to spawn one OS thread per thread in a block.
                                            // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                            // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                            #pragma omp parallel num_threads(static_cast<int>(uiNumThreadsInBlock))
                                            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                                // The first thread does some checks in the first block executed.
                                                if((::omp_get_thread_num() == 0) && (b == 0))
                                                {
                                                    auto const uiNumThreads(static_cast<decltype(uiNumThreadsInBlock)>(::omp_get_num_threads()));
                                                    printf("%s omp_get_num_threads: %d\n", BOOST_CURRENT_FUNCTION, uiNumThreads);
                                                    if(uiNumThreads != uiNumThreadsInBlock)
                                                    {
                                                        throw std::runtime_error("The OpenMP4 CPU runtime did not use the number of threads that had been required!");
                                                    }
                                                }
#endif
                                                std::forward<TKernelFunctor>(kernelFunctor)(
                                                    acc,
                                                    std::forward<TArgs>(args)...);

                                                // Wait for all threads to finish before deleting the shared memory.
                                                acc.syncBlockThreads();
                                            }

                                            // After a block has been processed, the shared memory has to be deleted.
                                            acc.m_vvuiSharedMem.clear();
                                        }
                                        // After all blocks have been processed, the external shared memory has to be deleted.
                                        acc.m_vuiExternalSharedMem.reset();
                                    }
                                }
                            }

                        public:
                            devs::cpu::detail::StreamCpu m_Stream;
                        };
                    }
                }
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The OpenMP4 CPU accelerator executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::omp::omp4::cpu::detail::ExecOmp4Cpu>
            {
                using type = accs::omp::omp4::cpu::detail::AccOmp4Cpu;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The OpenMP4 CPU accelerator executor event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::omp::omp4::cpu::detail::ExecOmp4Cpu>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The OpenMP4 CPU accelerator executor executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::omp::omp4::cpu::detail::ExecOmp4Cpu>
            {
                using type = accs::omp::omp4::cpu::detail::ExecOmp4Cpu;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The OpenMP4 CPU accelerator executor device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::omp::omp4::cpu::detail::ExecOmp4Cpu>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The OpenMP4 CPU accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::omp::omp4::cpu::detail::ExecOmp4Cpu>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The OpenMP4 CPU accelerator executor stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::omp::omp4::cpu::detail::ExecOmp4Cpu>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
            //#############################################################################
            //! The OpenMP4 CPU accelerator executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::omp::omp4::cpu::detail::ExecOmp4Cpu>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::omp::omp4::cpu::detail::ExecOmp4Cpu const & exec)
                -> devs::cpu::detail::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
