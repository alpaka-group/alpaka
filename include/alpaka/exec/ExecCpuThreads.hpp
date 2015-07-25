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
#include <alpaka/acc/Traits.hpp>                // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                // dev::traits::DevType
#include <alpaka/dim/Traits.hpp>                // dim::traits::DimType
#include <alpaka/event/Traits.hpp>              // event::traits::EventType
#include <alpaka/exec/Traits.hpp>               // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>               // size::traits::SizeType
#include <alpaka/stream/Traits.hpp>             // stream::traits::StreamType

// Implementation details.
#include <alpaka/acc/AccCpuThreads.hpp>         // acc:AccCpuThreads
#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu
#include <alpaka/event/EventCpuAsync.hpp>       // event::EventCpuAsync
#include <alpaka/kernel/Traits.hpp>             // kernel::getBlockSharedExternMemSizeBytes
#include <alpaka/stream/StreamCpuAsync.hpp>     // stream::StreamCpuAsync
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/core/ConcurrentExecPool.hpp>   // core::ConcurrentExecPool
#include <alpaka/core/NdLoop.hpp>               // core::NdLoop

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
    namespace exec
    {
        namespace threads
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU threads executor.
                //#############################################################################
                template<
                    typename TDim,
                    typename TSize>
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
                        ALPAKA_FN_HOST static auto yield()
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
                    using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
                        TSize,
                        std::thread,        // The concurrent execution type.
                        std::promise,       // The promise type.
                        ThreadPoolYield>;   // The type yielding the current concurrent execution.

                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ExecCpuThreadsImpl() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ExecCpuThreadsImpl(ExecCpuThreadsImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ExecCpuThreadsImpl(ExecCpuThreadsImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(ExecCpuThreadsImpl const &) -> ExecCpuThreadsImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(ExecCpuThreadsImpl &&) -> ExecCpuThreadsImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~ExecCpuThreadsImpl() = default;

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
                            dim::Dim<TWorkDiv>::value == TDim::value,
                            "The work division and the executor have to be of the same dimensionality!");

                        auto const vuiGridBlockExtents(
                            workdiv::getWorkDiv<Grid, Blocks>(workDiv));
                        auto const vuiBlockThreadExtents(
                            workdiv::getWorkDiv<Block, Threads>(workDiv));

                        auto const uiBlockSharedExternMemSizeBytes(
                            kernel::getBlockSharedExternMemSizeBytes<
                                typename std::decay<TKernelFnObj>::type,
                                acc::AccCpuThreads<TDim, TSize>>(
                                    vuiBlockThreadExtents,
                                    args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif
                        acc::AccCpuThreads<TDim, TSize> acc(workDiv);

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
                            &ExecCpuThreadsImpl<TDim, TSize>::gridBlockExecHost<TKernelFnObj, TArgs...>,
                            std::ref(acc),
                            std::placeholders::_1,
                            std::ref(vuiBlockThreadExtents),
                            std::ref(threadPool),
                            std::ref(kernelFnObj),
                            std::ref(args)...));

                        // Execute the blocks serially.
                        core::ndLoop(
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
                        typename TKernelFnObj,
                        typename... TArgs>
                    ALPAKA_FN_HOST static auto gridBlockExecHost(
                        acc::AccCpuThreads<TDim, TSize> & acc,
                        Vec<TDim, TSize> const & vuiGridBlockIdx,
                        Vec<TDim, TSize> const & vuiBlockThreadExtents,
                        ThreadPool & threadPool,
                        TKernelFnObj const & kernelFnObj,
                        TArgs const & ... args)
                    -> void
                    {
                         // The futures of the threads in the current block.
                        std::vector<std::future<void>> vFuturesInBlock;

                        // Set the index of the current block
                        acc.m_vuiGridBlockIdx = vuiGridBlockIdx;

                        // Bind the kernel and its arguments to the host block thread execution function.
                        auto boundBlockThreadExecHost(std::bind(
                            &ExecCpuThreadsImpl<TDim, TSize>::blockThreadExecHost<TKernelFnObj, TArgs...>,
                            std::ref(acc),
                            std::ref(vFuturesInBlock),
                            std::placeholders::_1,
                            std::ref(threadPool),
                            std::ref(kernelFnObj),
                            std::ref(args)...));
                        // Execute the block threads in parallel.
                        core::ndLoop(
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
                        block::shared::freeMem(acc);
                    }
                    //-----------------------------------------------------------------------------
                    //! The function executed for each block thread on the host.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFnObj,
                        typename... TArgs>
                    ALPAKA_FN_HOST static auto blockThreadExecHost(
                        acc::AccCpuThreads<TDim, TSize> & acc,
                        std::vector<std::future<void>> & vFuturesInBlock,
                        Vec<TDim, TSize> const & vuiBlockThreadIdx,
                        ThreadPool & threadPool,
                        TKernelFnObj const & kernelFnObj,
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
                                    kernelFnObj,
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
                        typename TKernelFnObj,
                        typename... TArgs>
                    ALPAKA_FN_HOST static auto blockThreadExecAcc(
                        acc::AccCpuThreads<TDim, TSize> & acc,
                        Vec<TDim, TSize> const & vuiBlockThreadIdx,
                        TKernelFnObj const & kernelFnObj,
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
                        typename std::map<std::thread::id, TSize>::iterator itThreadToBarrier;

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
                        kernelFnObj(
                            const_cast<acc::AccCpuThreads<TDim, TSize> const &>(acc),
                            args...);

                        // We have to sync all threads here because if a thread would finish before all threads have been started,
                        // a new thread could get the recycled (then duplicate) thread id!
                        acc.syncBlockThreads(itThreadToBarrier);
                    }
                };
            }
        }

        //#############################################################################
        //! The CPU threads executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class ExecCpuThreads final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuThreads(
                TWorkDiv const & workDiv,
                stream::StreamCpuAsync & stream) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    m_Stream(stream)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    dim::Dim<TWorkDiv>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuThreads(ExecCpuThreads const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuThreads(ExecCpuThreads &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuThreads const &) -> ExecCpuThreads & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuThreads &&) -> ExecCpuThreads & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ExecCpuThreads() = default;

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
                        threads::detail::ExecCpuThreadsImpl<TDim, TSize> exec;
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
            //! The CPU threads executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                exec::ExecCpuThreads<TDim, TSize>>
            {
                using type = acc::AccCpuThreads<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                exec::ExecCpuThreads<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU threads executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                exec::ExecCpuThreads<TDim, TSize>>
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
            //! The CPU threads executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                exec::ExecCpuThreads<TDim, TSize>>
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
            //! The CPU threads executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct EventType<
                exec::ExecCpuThreads<TDim, TSize>>
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
            //! The CPU threads executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                exec::ExecCpuThreads<TDim, TSize>>
            {
                using type = exec::ExecCpuThreads<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                exec::ExecCpuThreads<TDim, TSize>>
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
            //! The CPU threads executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct StreamType<
                exec::ExecCpuThreads<TDim, TSize>>
            {
                using type = stream::StreamCpuAsync;
            };
            //#############################################################################
            //! The CPU threads executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetStream<
                exec::ExecCpuThreads<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getStream(
                    exec::ExecCpuThreads<TDim, TSize> const & exec)
                -> stream::StreamCpuAsync
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
