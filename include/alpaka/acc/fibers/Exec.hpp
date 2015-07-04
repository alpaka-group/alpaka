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
#include <alpaka/acc/fibers/Acc.hpp>            // AccCpuFibers
#include <alpaka/dev/DevCpu.hpp>                // DevCpu
#include <alpaka/event/EventCpuAsync.hpp>       // EventCpuAsync
#include <alpaka/kernel/Traits.hpp>             // BlockSharedExternMemSizeBytes
#include <alpaka/stream/StreamCpuAsync.hpp>     // StreamCpuAsync
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/core/Fibers.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>   // ConcurrentExecPool
#include <alpaka/core/NdLoop.hpp>               // NdLoop

#include <boost/predef.h>                       // workarounds
#include <boost/align.hpp>                      // boost::aligned_alloc

#include <algorithm>                            // std::for_each
#include <vector>                               // std::vector
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace exec
    {
        namespace fibers
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU fibers accelerator executor implementation.
                //#############################################################################
                template<
                    typename TDim,
                    typename TSize>
                class ExecCpuFibersImpl final
                {
                private:
                    //#############################################################################
                    //! The type given to the ConcurrentExecPool for yielding the current fiber.
                    //#############################################################################
                    struct FiberPoolYield
                    {
                        //-----------------------------------------------------------------------------
                        //! Yields the current fiber.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_NO_CUDA static auto yield()
                        -> void
                        {
                            boost::this_fiber::yield();
                        }
                    };
                    //#############################################################################
                    // Yielding is not faster for fibers. Therefore we use condition variables.
                    // It is better to wake them up when the conditions are fulfilled because this does not cost as much as for real threads.
                    //#############################################################################
                    using FiberPool = alpaka::detail::ConcurrentExecPool<
                        TSize,
                        boost::fibers::fiber,               // The concurrent execution type.
                        boost::fibers::promise,             // The promise type.
                        FiberPoolYield,                     // The type yielding the current concurrent execution.
                        boost::fibers::mutex,               // The mutex type to use. Only required if TbYield is true.
                        boost::fibers::condition_variable,  // The condition variable type to use. Only required if TbYield is true.
                        false>;                             // If the threads should yield.

                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ExecCpuFibersImpl() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ExecCpuFibersImpl(ExecCpuFibersImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ExecCpuFibersImpl(ExecCpuFibersImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(ExecCpuFibersImpl const &) -> ExecCpuFibersImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(ExecCpuFibersImpl &&) -> ExecCpuFibersImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~ExecCpuFibersImpl() = default;

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
                                AccCpuFibers<TDim, TSize>>(
                                    vuiBlockThreadExtents,
                                    args...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif
                        AccCpuFibers<TDim, TSize> acc(workDiv);

                        if(uiBlockSharedExternMemSizeBytes > 0u)
                        {
                            acc.m_vuiExternalSharedMem.reset(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(16u, uiBlockSharedExternMemSizeBytes)));
                        }

                        auto const uiNumThreadsInBlock(vuiBlockThreadExtents.prod());
                        FiberPool fiberPool(uiNumThreadsInBlock, uiNumThreadsInBlock);

                        // Bind the kernel and its arguments to the grid block function.
                        auto boundGridBlockExecHost(std::bind(
                            &ExecCpuFibersImpl<TDim, TSize>::gridBlockExecHost<TKernelFnObj, TArgs...>,
                            std::ref(acc),
                            std::placeholders::_1,
                            std::ref(vuiBlockThreadExtents),
                            std::ref(fiberPool),
                            std::ref(kernelFnObj),
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
                        typename TKernelFnObj,
                        typename... TArgs>
                    ALPAKA_FN_HOST static auto gridBlockExecHost(
                        AccCpuFibers<TDim, TSize> & acc,
                        Vec<TDim, TSize> const & vuiGridBlockIdx,
                        Vec<TDim, TSize> const & vuiBlockThreadExtents,
                        FiberPool & fiberPool,
                        TKernelFnObj const & kernelFnObj,
                        TArgs const & ... args)
                    -> void
                    {
                         // The futures of the threads in the current block.
                        std::vector<boost::fibers::future<void>> vFuturesInBlock;

                        // Set the index of the current block
                        acc.m_vuiGridBlockIdx = vuiGridBlockIdx;

                        // Bind the kernel and its arguments to the host block thread execution function.
                        auto boundBlockThreadExecHost(std::bind(
                            &ExecCpuFibersImpl<TDim, TSize>::blockThreadExecHost<TKernelFnObj, TArgs...>,
                            std::ref(acc),
                            std::ref(vFuturesInBlock),
                            std::placeholders::_1,
                            std::ref(fiberPool),
                            std::ref(kernelFnObj),
                            std::ref(args)...));
                        // Execute the block threads in parallel.
                        ndLoop(
                            vuiBlockThreadExtents,
                            boundBlockThreadExecHost);

                        // Wait for the completion of the block thread kernels.
                        std::for_each(
                            vFuturesInBlock.begin(),
                            vFuturesInBlock.end(),
                            [](boost::fibers::future<void> & t)
                            {
                                t.wait();
                            }
                        );
                        // Clean up.
                        vFuturesInBlock.clear();

                        acc.m_mFibersToIndices.clear();
                        acc.m_mFibersToBarrier.clear();

                        // After a block has been processed, the shared memory has to be deleted.
                        block::shared::freeMem(acc);
                    }
                    //-----------------------------------------------------------------------------
                    //! The function executed for each block thread.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFnObj,
                        typename... TArgs>
                    ALPAKA_FN_HOST static auto blockThreadExecHost(
                        AccCpuFibers<TDim, TSize> & acc,
                        std::vector<boost::fibers::future<void>> & vFuturesInBlock,
                        Vec<TDim, TSize> const & vuiBlockThreadIdx,
                        FiberPool & fiberPool,
                        TKernelFnObj const & kernelFnObj,
                        TArgs const & ... args)
                    -> void
                    {
                        // Bind the arguments to the accelerator block thread execution function.
                        // The vuiBlockThreadIdx is required to be copied in because the variable will get changed for the next iteration/thread.
                        auto boundBlockThreadExecAcc(
                            [&, vuiBlockThreadIdx]()
                            {
                                blockThreadFiberFn(
                                    acc,
                                    vuiBlockThreadIdx,
                                    kernelFnObj,
                                    args...);
                            });
                        // Add the bound function to the block thread pool.
                        vFuturesInBlock.emplace_back(
                            fiberPool.enqueueTask(
                                boundBlockThreadExecAcc));
                    }
                    //-----------------------------------------------------------------------------
                    //! The fiber entry point.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFnObj,
                        typename... TArgs>
                    ALPAKA_FN_HOST static auto blockThreadFiberFn(
                        AccCpuFibers<TDim, TSize> & acc,
                        Vec<TDim, TSize> const & vuiBlockThreadIdx,
                        TKernelFnObj const & kernelFnObj,
                        TArgs const & ... args)
                    -> void
                    {
                        // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                        auto const idFiber(boost::this_fiber::get_id());

                        // Set the master thread id.
                        if(vuiBlockThreadIdx.sum() == 0)
                        {
                            acc.m_idMasterFiber = idFiber;
                        }

                        // We can not use the default syncBlockThreads here because it searches inside m_mFibersToBarrier for the thread id.
                        // Concurrently searching while others use emplace is unsafe!
                        typename std::map<boost::fibers::fiber::id, TSize>::iterator itFiberToBarrier;

                        // Save the fiber id, and index.
                        acc.m_mFibersToIndices.emplace(idFiber, vuiBlockThreadIdx);
                        itFiberToBarrier = acc.m_mFibersToBarrier.emplace(idFiber, 0).first;

                        // Sync all threads so that the maps with thread id's are complete and not changed after here.
                        acc.syncBlockThreads(itFiberToBarrier);

                        // Execute the kernel itself.
                        kernelFnObj(
                            const_cast<AccCpuFibers<TDim, TSize> const &>(acc),
                            args...);

                        // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recycled (then duplicate) fiber id!
                        acc.syncBlockThreads(itFiberToBarrier);
                    }
                };
            }
        }

        //#############################################################################
        //! The CPU fibers accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class ExecCpuFibers final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuFibers(
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
            ALPAKA_FN_HOST ExecCpuFibers(ExecCpuFibers const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuFibers(ExecCpuFibers &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuFibers const &) -> ExecCpuFibers & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuFibers &&) -> ExecCpuFibers & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ExecCpuFibers() = default;

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
                        fibers::detail::ExecCpuFibersImpl<TDim, TSize> exec;
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
            //! The CPU fibers executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                exec::ExecCpuFibers<TDim, TSize>>
            {
                using type = acc::fibers::detail::AccCpuFibers<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                exec::ExecCpuFibers<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU fibers executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                exec::ExecCpuFibers<TDim, TSize>>
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
            //! The CPU fibers executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                exec::ExecCpuFibers<TDim, TSize>>
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
            //! The CPU fibers executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct EventType<
                exec::ExecCpuFibers<TDim, TSize>>
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
            //! The CPU fibers executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                exec::ExecCpuFibers<TDim, TSize>>
            {
                using type = exec::ExecCpuFibers<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                exec::ExecCpuFibers<TDim, TSize>>
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
            //! The CPU fibers executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct StreamType<
                exec::ExecCpuFibers<TDim, TSize>>
            {
                using type = stream::StreamCpuAsync;
            };
            //#############################################################################
            //! The CPU fibers executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetStream<
                exec::ExecCpuFibers<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getStream(
                    exec::ExecCpuFibers<TDim, TSize> const & exec)
                -> stream::StreamCpuAsync
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
