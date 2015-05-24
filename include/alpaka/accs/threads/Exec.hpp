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
#include <alpaka/accs/threads/Acc.hpp>          // AccCpuThreads
#include <alpaka/core/BasicWorkDiv.hpp>         // WorkDivThreads
#include <alpaka/core/ConcurrentExecPool.hpp>   // ConcurrentExecPool
#include <alpaka/core/NdLoop.hpp>               // NdLoop
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
                //! The CPU threads executor.
                //#############################################################################
                template<
                    typename TDim>
                class ExecCpuThreads :
                    private AccCpuThreads<TDim>
                {
                private:
                    //#############################################################################
                    //! The type given to the ConcurrentExecPool for yielding the current thread.
                    //#############################################################################
                    struct ThreadPoolYield
                    {
                        //-----------------------------------------------------------------------------
                        //! Yields the current thread.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST static auto yield()
                        -> void
                        {
                            std::this_thread::yield();
                        }
                    };
                    //#############################################################################
                    // When using the thread pool the threads are yielding because this is faster.
                    // Using condition variables and going to sleep is very costly for real threads.
                    // Especially when the time to wait is really short (syncBlockThreads) yielding is much faster.
                    //#############################################################################
                    using ThreadPool = alpaka::detail::ConcurrentExecPool<
                        std::thread,                // The concurrent execution type.
                        std::promise,               // The promise type.
                        ThreadPoolYield>;           // The type yielding the current concurrent execution.

                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecCpuThreads(
                        TWorkDiv const & workDiv,
                        devs::cpu::StreamCpu & stream) :
                            AccCpuThreads<TDim>(workDiv),
                            m_Stream(stream),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuThreads(
                        ExecCpuThreads const & other) :
                            AccCpuThreads<TDim>(static_cast<workdiv::BasicWorkDiv<TDim> const &>(other)),
                            m_Stream(other.m_Stream),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuThreads(
                        ExecCpuThreads && other) :
                            AccCpuThreads<TDim>(static_cast<workdiv::BasicWorkDiv<TDim> &&>(other)),
                            m_Stream(other.m_Stream),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuThreads const &) -> ExecCpuThreads & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuThreads &&) -> ExecCpuThreads & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecCpuThreads() = default;
    #else
                    ALPAKA_FCT_HOST virtual ~ExecCpuThreads() noexcept = default;
    #endif

                    //-----------------------------------------------------------------------------
                    //! Enqueues the kernel functor.
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

                        m_Stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                            [this, kernelFunctor, args...]()
                            {
                                exec(
                                    kernelFunctor,
                                    args...);
                            });
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! Executes the kernel functor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto exec(
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const vuiGridBlockExtents(this->AccCpuThreads<TDim>::template getWorkDiv<Grid, Blocks>());
                        auto const vuiBlockThreadExtents(this->AccCpuThreads<TDim>::template getWorkDiv<Block, Threads>());

                        auto const uiBlockSharedExternMemSizeBytes(
                            kernel::getBlockSharedExternMemSizeBytes<
                                typename std::decay<TKernelFunctor>::type,
                                AccCpuThreads<TDim>>(
                                    vuiBlockThreadExtents,
                                    std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif
                        if(uiBlockSharedExternMemSizeBytes > 0)
                        {
                            this->AccCpuThreads<TDim>::m_vuiExternalSharedMem.reset(
                                new uint8_t[uiBlockSharedExternMemSizeBytes]);
                        }

                        auto const uiNumThreadsInBlock(vuiBlockThreadExtents.prod());
                        ThreadPool threadPool(uiNumThreadsInBlock, uiNumThreadsInBlock);

                        // Bind the kernel and its arguments to the grid block function.
                        auto boundGridBlockFct(std::bind(
                            &ExecCpuThreads<TDim>::gridBlockFct<TKernelFunctor&, TArgs&...>,
                            this,
                            std::placeholders::_1,
                            std::ref(vuiBlockThreadExtents),
                            std::ref(threadPool),
                            std::forward<TKernelFunctor>(kernelFunctor),
                            std::forward<TArgs>(args)...));

                        // Execute the blocks serially.
                        ndLoop(
                            vuiGridBlockExtents,
                            boundGridBlockFct);

                        // After all blocks have been processed, the external shared memory has to be deleted.
                        this->AccCpuThreads<TDim>::m_vuiExternalSharedMem.reset();
                    }
                    //-----------------------------------------------------------------------------
                    //! The function executed for each grid block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto gridBlockFct(
                        Vec<TDim> const & vuiGridBlockIdx,
                        Vec<TDim> const & vuiBlockThreadExtents,
                        ThreadPool & threadPool,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        this->AccCpuThreads<TDim>::m_vuiGridBlockIdx = vuiGridBlockIdx;
                        
                        // Bind the kernel and its arguments to the block thread function.
                        auto boundBlockThreadFct(std::bind(
                            &ExecCpuThreads<TDim>::blockThreadFct<TKernelFunctor&, TArgs&...>,
                            this,
                            std::placeholders::_1,
                            std::ref(threadPool),
                            std::forward<TKernelFunctor>(kernelFunctor),
                            std::forward<TArgs>(args)...));
                        // Execute the block threads in parallel.
                        ndLoop(
                            vuiBlockThreadExtents,
                            boundBlockThreadFct);

                        // Wait for the completion of the block thread kernels.
                        std::for_each(m_vFuturesInBlock.begin(), m_vFuturesInBlock.end(),
                            [](std::future<void> & t)
                            {
                                t.wait();
                            }
                        );
                        // Clean up.
                        m_vFuturesInBlock.clear();

                        this->AccCpuThreads<TDim>::m_mThreadsToIndices.clear();
                        this->AccCpuThreads<TDim>::m_mThreadsToBarrier.clear();

                        // After a block has been processed, the shared memory has to be deleted.
                        this->AccCpuThreads<TDim>::m_vvuiSharedMem.clear();
                    }
                    //-----------------------------------------------------------------------------
                    //! The function executed for each block thread.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto blockThreadFct(
                        Vec<TDim> const & vuiBlockThreadIdx,
                        ThreadPool & threadPool,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        // The vuiBlockThreadIdx is required to be copied in from the environment because if the thread is immediately suspended the variable is already changed for the next iteration/thread.
                        auto threadKernelFct =
                            [&, vuiBlockThreadIdx]()
                            {
                                blockThreadThreadFct(
                                    vuiBlockThreadIdx,
                                    std::forward<TKernelFunctor>(kernelFunctor),
                                    std::forward<TArgs>(args)...);
                            };
                        m_vFuturesInBlock.emplace_back(
                            threadPool.enqueueTask(
                                threadKernelFct));
                    }
                    //-----------------------------------------------------------------------------
                    //! The thread entry point.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto blockThreadThreadFct(
                        Vec<TDim> const & vuiBlockThreadIdx,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        // We have to store the thread data before the kernel is calling any of the methods of this class depending on them.
                        auto const idThread(std::this_thread::get_id());

                        // Set the master thread id.
                        if(vuiBlockThreadIdx.sum() == 0)
                        {
                            this->AccCpuThreads<TDim>::m_idMasterThread = idThread;
                        }

                        // We can not use the default syncBlockThreads here because it searches inside m_mThreadsToBarrier for the thread id.
                        // Concurrently searching while others use emplace is unsafe!
                        std::map<std::thread::id, UInt>::iterator itThreadToBarrier;

                        {
                            // The insertion of elements has to be done one thread at a time.
                            std::lock_guard<std::mutex> lock(m_mtxMapInsert);

                            // Save the thread id, and index.
                            this->AccCpuThreads<TDim>::m_mThreadsToIndices.emplace(idThread, vuiBlockThreadIdx);
                            itThreadToBarrier = this->AccCpuThreads<TDim>::m_mThreadsToBarrier.emplace(idThread, 0).first;
                        }

                        // Sync all fibers so that the maps with fiber id's are complete and not changed after here.
                        this->AccCpuThreads<TDim>::syncBlockThreads(itThreadToBarrier);

                        // Execute the kernel itself.
                        std::forward<TKernelFunctor>(kernelFunctor)(
                            (*static_cast<AccCpuThreads<TDim> const *>(this)),
                            std::forward<TArgs>(args)...);

                        // We have to sync all threads here because if a thread would finish before all threads have been started, the new thread could get a recycled (then duplicate) thread id!
                        this->AccCpuThreads<TDim>::syncBlockThreads(itThreadToBarrier);
                    }

                public:
                    devs::cpu::StreamCpu m_Stream;

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
            //! The CPU threads executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                using type = accs::threads::detail::AccCpuThreads<TDim>;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU threads executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU threads executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU threads executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                using type = TDim;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CPU threads executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct EventType<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                using type = devs::cpu::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU threads executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                using type = accs::threads::detail::ExecCpuThreads<TDim>;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CPU threads executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct StreamType<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                using type = devs::cpu::StreamCpu;
            };
            //#############################################################################
            //! The CPU threads executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetStream<
                accs::threads::detail::ExecCpuThreads<TDim>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::threads::detail::ExecCpuThreads<TDim> const & exec)
                -> devs::cpu::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
