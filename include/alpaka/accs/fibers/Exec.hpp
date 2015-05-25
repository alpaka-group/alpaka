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
#include <alpaka/accs/fibers/Acc.hpp>           // AccCpuFibers
#include <alpaka/accs/fibers/Common.hpp>
#include <alpaka/core/BasicWorkDiv.hpp>         // workdiv::BasicWorkDiv
#include <alpaka/core/ConcurrentExecPool.hpp>   // ConcurrentExecPool
#include <alpaka/core/NdLoop.hpp>               // NdLoop
#include <alpaka/devs/cpu/Dev.hpp>              // DevCpu
#include <alpaka/devs/cpu/Event.hpp>            // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>           // StreamCpu
#include <alpaka/traits/Kernel.hpp>             // BlockSharedExternMemSizeBytes

#include <boost/predef.h>                       // workarounds

#include <algorithm>                            // std::for_each
#include <utility>                              // std::forward
#include <vector>                               // std::vector
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace accs
    {
        namespace fibers
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU fibers accelerator executor.
                //#############################################################################
                template<
                    typename TDim>
                class ExecCpuFibers :
                    private AccCpuFibers<TDim>
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
                        ALPAKA_FCT_ACC_NO_CUDA static auto yield()
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
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecCpuFibers(
                        TWorkDiv const & workDiv,
                        devs::cpu::StreamCpu & stream) :
                            AccCpuFibers<TDim>(workDiv),
                            m_Stream(stream),
                            m_vFuturesInBlock()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuFibers(
                        ExecCpuFibers const & other) :
                            AccCpuFibers<TDim>(static_cast<workdiv::BasicWorkDiv<TDim> const &>(other)),
                            m_Stream(other.m_Stream),
                            m_vFuturesInBlock()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuFibers(
                        ExecCpuFibers && other) :
                            AccCpuFibers<TDim>(static_cast<workdiv::BasicWorkDiv<TDim> &&>(other)),
                            m_Stream(other.m_Stream),
                            m_vFuturesInBlock()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuFibers const &) -> ExecCpuFibers & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuFibers &&) -> ExecCpuFibers & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecCpuFibers() = default;
#else
                    ALPAKA_FCT_HOST virtual ~ExecCpuFibers() noexcept = default;
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

                        auto const vuiGridBlockExtents(this->AccCpuFibers<TDim>::template getWorkDiv<Grid, Blocks>());
                        auto const vuiBlockThreadExtents(this->AccCpuFibers<TDim>::template getWorkDiv<Block, Threads>());

                        auto const uiBlockSharedExternMemSizeBytes(
                            kernel::getBlockSharedExternMemSizeBytes<
                                typename std::decay<TKernelFunctor>::type,
                                AccCpuFibers<TDim>>(
                                    vuiBlockThreadExtents,
                                    std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif
                        if(uiBlockSharedExternMemSizeBytes > 0)
                        {
                            this->AccCpuFibers<TDim>::m_vuiExternalSharedMem.reset(
                                new uint8_t[uiBlockSharedExternMemSizeBytes]);
                        }

                        auto const uiNumThreadsInBlock(vuiBlockThreadExtents.prod());
                        FiberPool fiberPool(uiNumThreadsInBlock, uiNumThreadsInBlock);

                        // Bind the kernel and its arguments to the grid block function.
                        auto boundGridBlockFct(std::bind(
                            &ExecCpuFibers<TDim>::gridBlockFct<TKernelFunctor&, TArgs&...>,
                            this,
                            std::placeholders::_1,
                            std::ref(vuiBlockThreadExtents),
                            std::ref(fiberPool),
                            std::forward<TKernelFunctor>(kernelFunctor),
                            std::forward<TArgs>(args)...));

                        // Execute the blocks serially.
                        ndLoop(
                            vuiGridBlockExtents,
                            boundGridBlockFct);

                        // After all blocks have been processed, the external shared memory has to be deleted.
                        this->AccCpuFibers<TDim>::m_vuiExternalSharedMem.reset();
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
                        FiberPool & fiberPool,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        this->AccCpuFibers<TDim>::m_vuiGridBlockIdx = vuiGridBlockIdx;

                        // Bind the kernel and its arguments to the block thread function.
                        auto boundBlockThreadFct(std::bind(
                            &ExecCpuFibers<TDim>::blockThreadFct<TKernelFunctor&, TArgs&...>,
                            this,
                            std::placeholders::_1,
                            std::ref(fiberPool),
                            std::forward<TKernelFunctor>(kernelFunctor),
                            std::forward<TArgs>(args)...));
                        // Execute the block threads in parallel.
                        ndLoop(
                            vuiBlockThreadExtents,
                            boundBlockThreadFct);

                        // Wait for the completion of the block thread kernels.
                        std::for_each(m_vFuturesInBlock.begin(), m_vFuturesInBlock.end(),
                            [](boost::fibers::future<void> & t)
                            {
                                t.wait();
                            }
                        );
                        // Clean up.
                        m_vFuturesInBlock.clear();

                        this->AccCpuFibers<TDim>::m_mFibersToIndices.clear();
                        this->AccCpuFibers<TDim>::m_mFibersToBarrier.clear();

                        // After a block has been processed, the shared memory has to be deleted.
                        this->AccCpuFibers<TDim>::m_vvuiSharedMem.clear();
                    }
                    //-----------------------------------------------------------------------------
                    //! The function executed for each block thread.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto blockThreadFct(
                        Vec<TDim> const & vuiBlockThreadIdx,
                        FiberPool & fiberPool,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        // The vuiBlockThreadIdx is required to be copied in from the environment because if the thread is immediately suspended the variable is already changed for the next iteration/thread.
                        auto threadKernelFct =
                            [&, vuiBlockThreadIdx]()
                            {
                                blockThreadFiberFct(
                                    vuiBlockThreadIdx,
                                    std::forward<TKernelFunctor>(kernelFunctor),
                                    std::forward<TArgs>(args)...);
                            };
                        m_vFuturesInBlock.emplace_back(
                            fiberPool.enqueueTask(
                                threadKernelFct));
                    }
                    //-----------------------------------------------------------------------------
                    //! The fiber entry point.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto blockThreadFiberFct(
                        Vec<TDim> const & vuiBlockThreadIdx,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                        auto const idFiber(boost::this_fiber::get_id());

                        // Set the master thread id.
                        if(vuiBlockThreadIdx.sum() == 0)
                        {
                            this->AccCpuFibers<TDim>::m_idMasterFiber = idFiber;
                        }

                        // We can not use the default syncBlockThreads here because it searches inside m_mFibersToBarrier for the thread id.
                        // Concurrently searching while others use emplace is unsafe!
                        std::map<boost::fibers::fiber::id, UInt>::iterator itFiberToBarrier;

                        // Save the fiber id, and index.
                        this->AccCpuFibers<TDim>::m_mFibersToIndices.emplace(idFiber, vuiBlockThreadIdx);
                        itFiberToBarrier = this->AccCpuFibers<TDim>::m_mFibersToBarrier.emplace(idFiber, 0).first;

                        // Sync all threads so that the maps with thread id's are complete and not changed after here.
                        this->AccCpuFibers<TDim>::syncBlockThreads(itFiberToBarrier);

                        // Execute the kernel itself.
                        kernelFunctor(
                            (*static_cast<AccCpuFibers<TDim> const *>(this)),
                            std::forward<TArgs>(args)...);

                        // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recycled (then duplicate) fiber id!
                        this->AccCpuFibers<TDim>::syncBlockThreads(itFiberToBarrier);
                    }

                public:
                    devs::cpu::StreamCpu m_Stream;

                private:
                    std::vector<boost::fibers::future<void>> mutable m_vFuturesInBlock; //!< The futures of the fibers in the current block.
                };
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU fibers executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                using type = accs::fibers::detail::AccCpuFibers<TDim>;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU fibers executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU fibers executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU fibers executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                using type = TDim;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CPU fibers executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct EventType<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                using type = devs::cpu::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU fibers executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                using type = accs::fibers::detail::ExecCpuFibers<TDim>;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CPU fibers executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct StreamType<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                using type = devs::cpu::StreamCpu;
            };
            //#############################################################################
            //! The CPU fibers executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetStream<
                accs::fibers::detail::ExecCpuFibers<TDim>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::fibers::detail::ExecCpuFibers<TDim> const & exec)
                -> devs::cpu::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
