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
#include <alpaka/accs/threads/Acc.hpp>          // AccThreads
#include <alpaka/core/BasicWorkDiv.hpp>         // WorkDivThreads
#include <alpaka/core/ConcurrentExecPool.hpp>   // ConcurrentExecPool
#include <alpaka/devs/cpu/Dev.hpp>              // DevCpu
#include <alpaka/devs/cpu/Event.hpp>            // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>           // StreamCpu
#include <alpaka/traits/Kernel.hpp>             // BlockSharedExternMemSizeBytes

#include <boost/predef.h>                       // workarounds

#include <algorithm>                            // std::for_each
#include <stdexcept>                            // std::current_exception
#include <thread>                               // std::thread
#include <utility>                              // std::move, std::forward
#include <vector>                               // std::vector
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace accs
    {
        namespace threads
        {
            namespace detail
            {
                //#############################################################################
                //! The type given to the ConcurrentExecPool for yielding the current thread.
                //#############################################################################
                struct ThreadPoolYield
                {
                    //-----------------------------------------------------------------------------
                    //! Yields the current thread.
                    //-----------------------------------------------------------------------------
                    static auto yield()
                    -> void
                    {
                        std::this_thread::yield();
                    }
                };
                //#############################################################################
                //! The threads accelerator executor.
                //#############################################################################
                class ExecThreads :
                    private AccThreads
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecThreads(
                        TWorkDiv const & workDiv,
                        devs::cpu::detail::StreamCpu & stream) :
                            AccThreads(workDiv),
                            m_Stream(stream),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecThreads(
                        ExecThreads const & other) :
                            AccThreads(static_cast<workdiv::BasicWorkDiv const &>(other)),
                            m_Stream(other.m_Stream),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecThreads(
                        ExecThreads && other) :
                            AccThreads(static_cast<workdiv::BasicWorkDiv &&>(other)),
                            m_Stream(other.m_Stream),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecThreads const &) -> ExecThreads & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecThreads() = default;
    #else
                    ALPAKA_FCT_HOST virtual ~ExecThreads() noexcept = default;
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

                        Vec3<> const v3uiGridBlockExtents(this->AccThreads::getWorkDiv<Grid, Blocks, dim::Dim3>());
                        Vec3<> const v3uiBlockThreadExtents(this->AccThreads::getWorkDiv<Block, Threads, dim::Dim3>());

                        auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccThreads>(
                            v3uiBlockThreadExtents,
                            std::forward<TArgs>(args)...));
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
    #endif
                        if(uiBlockSharedExternMemSizeBytes > 0)
                        {
                            this->AccThreads::m_vuiExternalSharedMem.reset(
                                new uint8_t[uiBlockSharedExternMemSizeBytes]);
                        }

                        auto const uiNumThreadsInBlock(this->AccThreads::getWorkDiv<Block, Threads, dim::Dim1>());
                        // When using the thread pool the threads are yielding because this is faster.
                        // Using condition variables and going to sleep is very costly for real threads.
                        // Especially when the time to wait is really short (syncBlockThreads) yielding is much faster.
                        using ThreadPool = alpaka::detail::ConcurrentExecPool<
                            std::thread,                // The concurrent execution type.
                            std::promise,               // The promise type.
                            ThreadPoolYield>;           // The type yielding the current concurrent execution.
                        ThreadPool pool(uiNumThreadsInBlock[0u], uiNumThreadsInBlock[0u]);

                        // Execute the blocks serially.
                        for(this->AccThreads::m_v3uiGridBlockIdx[0u] = 0u; this->AccThreads::m_v3uiGridBlockIdx[0u]<v3uiGridBlockExtents[0u]; ++this->AccThreads::m_v3uiGridBlockIdx[0u])
                        {
                            for(this->AccThreads::m_v3uiGridBlockIdx[1u] = 0u; this->AccThreads::m_v3uiGridBlockIdx[1u]<v3uiGridBlockExtents[1u]; ++this->AccThreads::m_v3uiGridBlockIdx[1u])
                            {
                                for(this->AccThreads::m_v3uiGridBlockIdx[2u] = 0u; this->AccThreads::m_v3uiGridBlockIdx[2u]<v3uiGridBlockExtents[2u]; ++this->AccThreads::m_v3uiGridBlockIdx[2u])
                                {
                                    // Execute the threads in parallel.
                                    Vec3<> v3uiBlockThreadIdx(Vec3<>::zeros());
                                    for(v3uiBlockThreadIdx[0u] = 0u; v3uiBlockThreadIdx[0u]<v3uiBlockThreadExtents[0u]; ++v3uiBlockThreadIdx[0u])
                                    {
                                        for(v3uiBlockThreadIdx[1u] = 0u; v3uiBlockThreadIdx[1u]<v3uiBlockThreadExtents[1u]; ++v3uiBlockThreadIdx[1u])
                                        {
                                            for(v3uiBlockThreadIdx[2u] = 0u; v3uiBlockThreadIdx[2u]<v3uiBlockThreadExtents[2u]; ++v3uiBlockThreadIdx[2u])
                                            {
                                                // The v3uiBlockThreadIdx is required to be copied in from the environment because if the thread is immediately suspended the variable is already changed for the next iteration/thread.
                                                auto threadKernelFct =
                                                    [&, v3uiBlockThreadIdx]()
                                                    {
                                                        threadKernel(
                                                            v3uiBlockThreadIdx,
                                                            std::forward<TKernelFunctor>(kernelFunctor),
                                                            std::forward<TArgs>(args)...);
                                                    };
                                                m_vFuturesInBlock.emplace_back(
                                                    pool.enqueueTask(
                                                        threadKernelFct));
                                            }
                                        }
                                    }

                                    // Wait for the completion of the kernels.
                                    std::for_each(m_vFuturesInBlock.begin(), m_vFuturesInBlock.end(),
                                        [](std::future<void> & t)
                                        {
                                            t.wait();
                                        }
                                    );
                                    // Clean up.
                                    m_vFuturesInBlock.clear();

                                    this->AccThreads::m_mThreadsToIndices.clear();
                                    this->AccThreads::m_mThreadsToBarrier.clear();

                                    // After a block has been processed, the shared memory has to be deleted.
                                    this->AccThreads::m_vvuiSharedMem.clear();
                                }
                            }
                        }
                        // After all blocks have been processed, the external shared memory has to be deleted.
                        this->AccThreads::m_vuiExternalSharedMem.reset();
                    }
                private:
                    //-----------------------------------------------------------------------------
                    //! The thread entry point.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto threadKernel(
                        Vec3<> const & v3uiBlockThreadIdx,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        // We have to store the thread data before the kernel is calling any of the methods of this class depending on them.
                        auto const idThread(std::this_thread::get_id());

                        // Set the master thread id.
                        if(v3uiBlockThreadIdx[0] == 0 && v3uiBlockThreadIdx[1] == 0 && v3uiBlockThreadIdx[2] == 0)
                        {
                            this->AccThreads::m_idMasterThread = idThread;
                        }

                        // We can not use the default syncBlockThreads here because it searches inside m_mThreadsToBarrier for the thread id.
                        // Concurrently searching while others use emplace is unsafe!
                        std::map<std::thread::id, UInt>::iterator itThreadToBarrier;

                        {
                            // The insertion of elements has to be done one thread at a time.
                            std::lock_guard<std::mutex> lock(m_mtxMapInsert);

                            // Save the thread id, and index.
                            this->AccThreads::m_mThreadsToIndices.emplace(idThread, v3uiBlockThreadIdx);
                            itThreadToBarrier = this->AccThreads::m_mThreadsToBarrier.emplace(idThread, 0).first;
                        }

                        // Sync all fibers so that the maps with fiber id's are complete and not changed after here.
                        this->AccThreads::syncBlockThreads(itThreadToBarrier);

                        // Execute the kernel itself.
                        std::forward<TKernelFunctor>(kernelFunctor)(
                            (*static_cast<AccThreads const *>(this)),
                            std::forward<TArgs>(args)...);

                        // We have to sync all threads here because if a thread would finish before all threads have been started, the new thread could get a recycled (then duplicate) thread id!
                        this->AccThreads::syncBlockThreads(itThreadToBarrier);
                    }

                public:
                    devs::cpu::detail::StreamCpu m_Stream;

                private:
                    std::vector<std::future<void>> mutable m_vFuturesInBlock; //!< The futures of the threads in the current block.

                    std::mutex mutable m_mtxMapInsert;
                };
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The threads accelerator executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::threads::detail::ExecThreads>
            {
                using type = accs::threads::detail::AccThreads;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The threads accelerator executor event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::threads::detail::ExecThreads>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The threads accelerator executor executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::threads::detail::ExecThreads>
            {
                using type = accs::threads::detail::ExecThreads;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The threads accelerator executor device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::threads::detail::ExecThreads>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The threads accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::threads::detail::ExecThreads>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The threads accelerator executor stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::threads::detail::ExecThreads>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
            //#############################################################################
            //! The threads accelerator executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::threads::detail::ExecThreads>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::threads::detail::ExecThreads const & exec)
                -> devs::cpu::detail::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
