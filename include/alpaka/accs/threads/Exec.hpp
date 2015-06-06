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
#include <boost/align.hpp>                      // boost::aligned_alloc

#include <algorithm>                            // std::for_each
#include <thread>                               // std::thread
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
                class ExecCpuThreadsImpl final
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
                        std::thread,        // The concurrent execution type.
                        std::promise,       // The promise type.
                        ThreadPoolYield>;   // The type yielding the current concurrent execution.

                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuThreadsImpl() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuThreadsImpl(ExecCpuThreadsImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuThreadsImpl(ExecCpuThreadsImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuThreadsImpl const &) -> ExecCpuThreadsImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuThreadsImpl &&) -> ExecCpuThreadsImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~ExecCpuThreadsImpl() noexcept = default;

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
                                AccCpuThreads<TDim>>(
                                    vuiBlockThreadExtents,
                                    args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif
                        AccCpuThreads<TDim> acc(workDiv);

                        if(uiBlockSharedExternMemSizeBytes > 0u)
                        {
                            acc.m_vuiExternalSharedMem.reset(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(16u, uiBlockSharedExternMemSizeBytes)));
                        }

                        auto const uiNumThreadsInBlock(vuiBlockThreadExtents.prod());
                        ThreadPool threadPool(uiNumThreadsInBlock, uiNumThreadsInBlock);

                        // Bind the kernel and its arguments to the grid block function.
                        auto boundGridBlockExecHost(std::bind(
                            &ExecCpuThreadsImpl<TDim>::gridBlockExecHost<TKernelFunctor, TArgs...>,
                            std::ref(acc),
                            std::placeholders::_1,
                            std::ref(vuiBlockThreadExtents),
                            std::ref(threadPool),
                            std::ref(kernelFunctor),
                            std::ref(args)...));

                        // Execute the blocks serially.
                        ndLoop(
                            vuiGridBlockExtents,
                            boundGridBlockExecHost);

                        // After all blocks have been processed, the external shared memory has to be deleted.
                        acc.m_vuiExternalSharedMem.reset();
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! The function executed for each grid block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST static auto gridBlockExecHost(
                        AccCpuThreads<TDim> & acc,
                        Vec<TDim> const & vuiGridBlockIdx,
                        Vec<TDim> const & vuiBlockThreadExtents,
                        ThreadPool & threadPool,
                        TKernelFunctor const & kernelFunctor,
                        TArgs const & ... args)
                    -> void
                    {
                         // The futures of the threads in the current block.
                        std::vector<std::future<void>> vFuturesInBlock;

                        // Set the index of the current block
                        acc.m_vuiGridBlockIdx = vuiGridBlockIdx;

                        // Bind the kernel and its arguments to the host block thread execution function.
                        auto boundBlockThreadExecHost(std::bind(
                            &ExecCpuThreadsImpl<TDim>::blockThreadExecHost<TKernelFunctor, TArgs...>,
                            std::ref(acc),
                            std::ref(vFuturesInBlock),
                            std::placeholders::_1,
                            std::ref(threadPool),
                            std::ref(kernelFunctor),
                            std::ref(args)...));
                        // Execute the block threads in parallel.
                        ndLoop(
                            vuiBlockThreadExtents,
                            boundBlockThreadExecHost);

                        // Wait for the completion of the block thread kernels.
                        std::for_each(
                            vFuturesInBlock.begin(),
                            vFuturesInBlock.end(),
                            [](std::future<void> & t)
                            {
                                t.wait();
                            }
                        );
                        // Clean up.
                        vFuturesInBlock.clear();

                        acc.m_mThreadsToIndices.clear();
                        acc.m_mThreadsToBarrier.clear();

                        // After a block has been processed, the shared memory has to be deleted.
                        acc.m_vvuiSharedMem.clear();
                    }
                    //-----------------------------------------------------------------------------
                    //! The function executed for each block thread on the host.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST static auto blockThreadExecHost(
                        AccCpuThreads<TDim> & acc,
                        std::vector<std::future<void>> & vFuturesInBlock,
                        Vec<TDim> const & vuiBlockThreadIdx,
                        ThreadPool & threadPool,
                        TKernelFunctor const & kernelFunctor,
                        TArgs const & ... args)
                    -> void
                    {
                        // Bind the arguments to the accelerator block thread execution function.
                        // The vuiBlockThreadIdx is required to be copied in because the variable will get changed for the next iteration/thread.
                        auto boundBlockThreadExecAcc(
                            [&, vuiBlockThreadIdx]()
                            {
                                blockThreadExecAcc(
                                    acc,
                                    vuiBlockThreadIdx,
                                    kernelFunctor,
                                    args...);
                            });
                        // Add the bound function to the block thread pool.
                        vFuturesInBlock.emplace_back(
                            threadPool.enqueueTask(
                                boundBlockThreadExecAcc));
                    }
                    //-----------------------------------------------------------------------------
                    //! The thread entry point on the accelerator.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST static auto blockThreadExecAcc(
                        AccCpuThreads<TDim> & acc,
                        Vec<TDim> const & vuiBlockThreadIdx,
                        TKernelFunctor const & kernelFunctor,
                        TArgs const & ... args)
                    -> void
                    {
                        // We have to store the thread data before the kernel is calling any of the methods of this class depending on them.
                        auto const idThread(std::this_thread::get_id());

                        // Set the master thread id.
                        if(vuiBlockThreadIdx.sum() == 0)
                        {
                            acc.m_idMasterThread = idThread;
                        }

                        // We can not use the default syncBlockThreads here because it searches inside m_mThreadsToBarrier for the thread id.
                        // Concurrently searching while others use emplace is unsafe!
                        std::map<std::thread::id, UInt>::iterator itThreadToBarrier;

                        {
                            // The insertion of elements has to be done one thread at a time.
                            std::lock_guard<std::mutex> lock(acc.m_mtxMapInsert);

                            // Save the thread id, and index.
                            acc.m_mThreadsToIndices.emplace(idThread, vuiBlockThreadIdx);
                            itThreadToBarrier = acc.m_mThreadsToBarrier.emplace(idThread, 0).first;
                        }

                        // Sync all threads so that the maps with thread id's are complete and not changed after here.
                        acc.syncBlockThreads(itThreadToBarrier);

                        // Execute the kernel itself.
                        kernelFunctor(
                            acc,
                            args...);

                        // We have to sync all threads here because if a thread would finish before all threads have been started, 
                        // a new thread could get the recycled (then duplicate) thread id!
                        acc.syncBlockThreads(itThreadToBarrier);
                    }
                };

                //#############################################################################
                //! The CPU threads executor.
                //#############################################################################
                template<
                    typename TDim>
                class ExecCpuThreads final :
                    public workdiv::BasicWorkDiv<TDim>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecCpuThreads(
                        TWorkDiv const & workDiv,
                        devs::cpu::StreamCpu & stream) :
                            workdiv::BasicWorkDiv<TDim>(workDiv),
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
                    ALPAKA_FCT_HOST ExecCpuThreads(ExecCpuThreads const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuThreads(ExecCpuThreads &&) = default;
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
                    ALPAKA_FCT_HOST ~ExecCpuThreads() noexcept = default;

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
                                ExecCpuThreadsImpl<TDim> exec;
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
