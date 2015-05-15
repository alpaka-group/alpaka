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
#include <alpaka/accs/omp/omp2/Acc.hpp>         // AccOmp2
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
            namespace omp2
            {
                namespace detail
                {
                    //#############################################################################
                    //! The OpenMP2 accelerator executor.
                    //#############################################################################
                    class ExecOmp2 :
                        private AccOmp2
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TWorkDiv>
                        ALPAKA_FCT_HOST ExecOmp2(
                            TWorkDiv const & workDiv,
                            devs::cpu::detail::StreamCpu & stream) :
                                AccOmp2(workDiv),
                                m_Stream(stream)
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        }
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecOmp2(
                            ExecOmp2 const & other) :
                                AccOmp2(static_cast<workdiv::BasicWorkDiv const &>(other)),
                                m_Stream(other.m_Stream)
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecOmp2(
                            ExecOmp2 && other) :
                                AccOmp2(static_cast<workdiv::BasicWorkDiv &&>(other)),
                                m_Stream(other.m_Stream)
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        }
#endif
                        //-----------------------------------------------------------------------------
                        //! Copy assignment.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator=(ExecOmp2 const &) -> ExecOmp2 & = delete;
                        //-----------------------------------------------------------------------------
                        //! Destructor.
                        //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                        ALPAKA_FCT_HOST virtual ~ExecOmp2() = default;
#else
                        ALPAKA_FCT_HOST virtual ~ExecOmp2() noexcept = default;
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

                            Vec3<> const v3uiGridBlockExtents(this->AccOmp2::getWorkDiv<Grid, Blocks, dim::Dim3>());
                            Vec3<> const v3uiBlockThreadExtents(this->AccOmp2::getWorkDiv<Block, Threads, dim::Dim3>());

                            auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccOmp2>(
                                v3uiBlockThreadExtents,
                                std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                                << std::endl;
#endif
                            this->AccOmp2::m_vuiExternalSharedMem.reset(
                                new uint8_t[uiBlockSharedExternMemSizeBytes]);

                            // The number of threads in this block.
                            auto const uiNumThreadsInBlock(this->AccOmp2::getWorkDiv<Block, Threads, dim::Dim1>()[0]);

                            // Execute the blocks serially.
                            for(this->AccOmp2::m_v3uiGridBlockIdx[0u] = 0u; this->AccOmp2::m_v3uiGridBlockIdx[0u]<v3uiGridBlockExtents[0u]; ++this->AccOmp2::m_v3uiGridBlockIdx[0u])
                            {
                                for(this->AccOmp2::m_v3uiGridBlockIdx[1u] = 0u; this->AccOmp2::m_v3uiGridBlockIdx[1u]<v3uiGridBlockExtents[1u]; ++this->AccOmp2::m_v3uiGridBlockIdx[1u])
                                {
                                    for(this->AccOmp2::m_v3uiGridBlockIdx[2u] = 0u; this->AccOmp2::m_v3uiGridBlockIdx[2u]<v3uiGridBlockExtents[2u]; ++this->AccOmp2::m_v3uiGridBlockIdx[2u])
                                    {
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
                                            if((::omp_get_thread_num() == 0) && (this->AccOmp2::m_v3uiGridBlockIdx[0u] == 0u) && (this->AccOmp2::m_v3uiGridBlockIdx[1u] == 0u) && (this->AccOmp2::m_v3uiGridBlockIdx[2u] == 0u))
                                            {
                                                auto const uiNumThreads(static_cast<decltype(uiNumThreadsInBlock)>(::omp_get_num_threads()));
                                                std::cout << BOOST_CURRENT_FUNCTION << " omp_get_num_threads: " << uiNumThreads << std::endl;
                                                if(uiNumThreads != uiNumThreadsInBlock)
                                                {
                                                    throw std::runtime_error("The OpenMP2 runtime did not use the number of threads that had been required!");
                                                }
                                            }
#endif
                                            std::forward<TKernelFunctor>(kernelFunctor)(
                                                (*static_cast<AccOmp2 const *>(this)),
                                                std::forward<TArgs>(args)...);

                                            // Wait for all threads to finish before deleting the shared memory.
                                            this->AccOmp2::syncBlockThreads();
                                        }

                                        // After a block has been processed, the shared memory has to be deleted.
                                        this->AccOmp2::m_vvuiSharedMem.clear();
                                    }
                                }
                            }
                            // After all blocks have been processed, the external shared memory has to be deleted.
                            this->AccOmp2::m_vuiExternalSharedMem.reset();
                        }

                    public:
                        devs::cpu::detail::StreamCpu m_Stream;
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
            //! The OpenMP2 accelerator executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::omp::omp2::detail::ExecOmp2>
            {
                using type = accs::omp::omp2::detail::AccOmp2;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The OpenMP2 accelerator executor event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::omp::omp2::detail::ExecOmp2>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The OpenMP2 accelerator executor executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::omp::omp2::detail::ExecOmp2>
            {
                using type = accs::omp::omp2::detail::ExecOmp2;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The OpenMP2 accelerator executor device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::omp::omp2::detail::ExecOmp2>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The OpenMP2 accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::omp::omp2::detail::ExecOmp2>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The OpenMP2 accelerator executor stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::omp::omp2::detail::ExecOmp2>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
            //#############################################################################
            //! The OpenMP2 accelerator executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::omp::omp2::detail::ExecOmp2>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::omp::omp2::detail::ExecOmp2 const & exec)
                -> devs::cpu::detail::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
