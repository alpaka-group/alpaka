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
#include <alpaka/exec/Traits.hpp>               // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>               // size::traits::SizeType

// Implementation details.
#include <alpaka/acc/AccCpuFibers.hpp>          // acc:AccCpuFibers
#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu
#include <alpaka/kernel/Traits.hpp>             // kernel::getBlockSharedExternMemSizeBytes
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/core/Fibers.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>   // core::ConcurrentExecPool
#include <alpaka/core/NdLoop.hpp>               // core::NdLoop
#include <alpaka/core/ApplyTuple.hpp>           // core::Apply

#include <boost/predef.h>                       // workarounds
#include <boost/align.hpp>                      // boost::aligned_alloc

#include <algorithm>                            // std::for_each
#include <vector>                               // std::vector
#include <tuple>                                // std::tuple
#include <type_traits>                          // std::decay
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace exec
    {
        //#############################################################################
        //! The CPU fibers accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuFibers final :
            public workdiv::WorkDivMembers<TDim, TSize>
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
            using FiberPool = alpaka::core::detail::ConcurrentExecPool<
                TSize,
                boost::fibers::fiber,               // The concurrent execution type.
                boost::fibers::promise,             // The promise type.
                FiberPoolYield,                     // The type yielding the current concurrent execution.
                boost::fibers::mutex,               // The mutex type to use. Only required if TisYielding is true.
                boost::fibers::condition_variable,  // The condition variable type to use. Only required if TisYielding is true.
                false>;                             // If the threads should yield.

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuFibers(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TSize>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
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
            //! Executes the kernel function object.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator()() const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtents(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));
                auto const blockThreadExtents(
                    workdiv::getWorkDiv<Block, Threads>(*this));

                // Get the size of the block shared extern memory.
                auto const blockSharedExternMemSizeBytes(
                    core::apply(
                        [&](TArgs const & ... args)
                        {
                            return
                                kernel::getBlockSharedExternMemSizeBytes<
                                    TKernelFnObj,
                                    acc::AccCpuFibers<TDim, TSize>>(
                                        blockThreadExtents,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION
                    << " BlockSharedExternMemSizeBytes: " << blockSharedExternMemSizeBytes << " B" << std::endl;
#endif
                acc::AccCpuFibers<TDim, TSize> acc(*static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this));

                if(blockSharedExternMemSizeBytes > 0u)
                {
                    acc.m_externalSharedMem.reset(
                        reinterpret_cast<uint8_t *>(
                            boost::alignment::aligned_alloc(16u, blockSharedExternMemSizeBytes)));
                }

                auto const numThreadsInBlock(blockThreadExtents.prod());
                FiberPool fiberPool(numThreadsInBlock, numThreadsInBlock);

                auto const boundGridBlockExecHost(
                    core::apply(
                        [this, &acc, &blockThreadExtents, &fiberPool](TArgs const & ... args)
                        {
                            // Bind the kernel and its arguments to the grid block function.
                            return
                                std::bind(
                                    &ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>::gridBlockExecHost,
                                    std::ref(acc),
                                    std::placeholders::_1,
                                    std::ref(blockThreadExtents),
                                    std::ref(fiberPool),
                                    std::ref(m_kernelFnObj),
                                    std::ref(args)...);
                        },
                        m_args));

                // Execute the blocks serially.
                core::ndLoopIncIdx(
                    gridBlockExtents,
                    boundGridBlockExecHost);

                // After all blocks have been processed, the external shared memory has to be deleted.
                acc.m_externalSharedMem.reset();
            }

        private:
            //-----------------------------------------------------------------------------
            //! The function executed for each grid block.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto gridBlockExecHost(
                acc::AccCpuFibers<TDim, TSize> & acc,
                Vec<TDim, TSize> const & gridBlockIdx,
                Vec<TDim, TSize> const & blockThreadExtents,
                FiberPool & fiberPool,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
            -> void
            {
                    // The futures of the threads in the current block.
                std::vector<boost::fibers::future<void>> futuresInBlock;

                // Set the index of the current block
                acc.m_gridBlockIdx = gridBlockIdx;

                // Bind the kernel and its arguments to the host block thread execution function.
                auto boundBlockThreadExecHost(std::bind(
                    &ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>::blockThreadExecHost,
                    std::ref(acc),
                    std::ref(futuresInBlock),
                    std::placeholders::_1,
                    std::ref(fiberPool),
                    std::ref(kernelFnObj),
                    std::ref(args)...));
                // Execute the block threads in parallel.
                core::ndLoopIncIdx(
                    blockThreadExtents,
                    boundBlockThreadExecHost);

                // Wait for the completion of the block thread kernels.
                std::for_each(
                    futuresInBlock.begin(),
                    futuresInBlock.end(),
                    [](boost::fibers::future<void> & t)
                    {
                        t.wait();
                    }
                );
                // Clean up.
                futuresInBlock.clear();

                acc.m_fibersToIndices.clear();
                acc.m_fibersToBarrier.clear();

                // After a block has been processed, the shared memory has to be deleted.
                block::shared::freeMem(acc);
            }
            //-----------------------------------------------------------------------------
            //! The function executed for each block thread.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto blockThreadExecHost(
                acc::AccCpuFibers<TDim, TSize> & acc,
                std::vector<boost::fibers::future<void>> & futuresInBlock,
                Vec<TDim, TSize> const & blockThreadIdx,
                FiberPool & fiberPool,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
            -> void
            {
                // Bind the arguments to the accelerator block thread execution function.
                // The blockThreadIdx is required to be copied in because the variable will get changed for the next iteration/thread.
                auto boundBlockThreadExecAcc(
                    [&, blockThreadIdx]()
                    {
                        blockThreadFiberFn(
                            acc,
                            blockThreadIdx,
                            kernelFnObj,
                            args...);
                    });
                // Add the bound function to the block thread pool.
                futuresInBlock.emplace_back(
                    fiberPool.enqueueTask(
                        boundBlockThreadExecAcc));
            }
            //-----------------------------------------------------------------------------
            //! The fiber entry point.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto blockThreadFiberFn(
                acc::AccCpuFibers<TDim, TSize> & acc,
                Vec<TDim, TSize> const & blockThreadIdx,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
            -> void
            {
                // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                auto const fiberId(boost::this_fiber::get_id());

                // Set the master thread id.
                if(blockThreadIdx.sum() == 0)
                {
                    acc.m_masterFiberId = fiberId;
                }

                // We can not use the default syncBlockThreads here because it searches inside m_fibersToBarrier for the thread id.
                // Concurrently searching while others use emplace is unsafe!
                typename std::map<boost::fibers::fiber::id, TSize>::iterator itFiberToBarrier;

                // Save the fiber id, and index.
                acc.m_fibersToIndices.emplace(fiberId, blockThreadIdx);
                itFiberToBarrier = acc.m_fibersToBarrier.emplace(fiberId, 0).first;

                // Sync all threads so that the maps with thread id's are complete and not changed after here.
                acc.syncBlockThreads(itFiberToBarrier);

                // Execute the kernel itself.
                kernelFnObj(
                    const_cast<acc::AccCpuFibers<TDim, TSize> const &>(acc),
                    args...);

                // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recycled (then duplicate) fiber id!
                acc.syncBlockThreads(itFiberToBarrier);
            }

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuFibers<TDim, TSize>;
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU fibers executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevManType<
                exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>>
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>;
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}
