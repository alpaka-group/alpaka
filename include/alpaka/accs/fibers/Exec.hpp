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
#include <alpaka/accs/fibers/Acc.hpp>           // AccFibers
#include <alpaka/accs/fibers/Common.hpp>
#include <alpaka/core/BasicWorkDiv.hpp>         // workdiv::BasicWorkDiv
#include <alpaka/core/ConcurrentExecPool.hpp>   // ConcurrentExecPool
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
                //! The type given to the ConcurrentExecPool for returning the current exception.
                //#############################################################################
                struct FiberPoolCurrentException
                {
                    //-----------------------------------------------------------------------------
                    //! \return The current exception.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA static auto current_exception()
                    -> std::result_of<decltype(&boost::current_exception)()>::type
                    {
                        return boost::current_exception();
                    }
                };

                //#############################################################################
                //! The fibers accelerator executor.
                //#############################################################################
                class ExecFibers :
                    private AccFibers
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecFibers(
                        TWorkDiv const & workDiv,
                        devs::cpu::detail::StreamCpu & stream):
                            AccFibers(workDiv),
                            m_Stream(stream),
                            m_vFuturesInBlock()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecFibers(
                        ExecFibers const & other):
                            AccFibers(static_cast<workdiv::BasicWorkDiv const &>(other)),
                            m_Stream(other.m_Stream),
                            m_vFuturesInBlock()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecFibers(
                        ExecFibers && other) :
                            AccFibers(static_cast<workdiv::BasicWorkDiv &&>(other)),
                            m_Stream(other.m_Stream)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecFibers const &) -> ExecFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecFibers() = default;
#else
                    ALPAKA_FCT_HOST virtual ~ExecFibers() noexcept = default;
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

                        Vec3<> const v3uiGridBlockExtents(this->AccFibers::getWorkDiv<Grid, Blocks, dim::Dim3>());
                        Vec3<> const v3uiBlockThreadExtents(this->AccFibers::getWorkDiv<Block, Threads, dim::Dim3>());

                        auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccFibers>(
                            v3uiBlockThreadExtents,
                            std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
#endif
                        this->AccFibers::m_vuiExternalSharedMem.reset(
                            new uint8_t[uiBlockSharedExternMemSizeBytes]);

                        auto const uiNumThreadsInBlock(this->AccFibers::getWorkDiv<Block, Threads, dim::Dim1>());
                        // Yielding is not faster for fibers. Therefore we use condition variables.
                        // It is better to wake them up when the conditions are fulfilled because this does not cost as much as for real threads.
                        using FiberPool = alpaka::detail::ConcurrentExecPool<
                            boost::fibers::fiber,               // The concurrent execution type.
                            boost::fibers::promise,             // The promise type.
                            FiberPoolCurrentException,          // The type returning the current exception.
                            FiberPoolYield,                     // The type yielding the current concurrent execution.
                            boost::fibers::mutex,               // The mutex type to use. Only required if TbYield is true.
                            boost::unique_lock,                 // The unique lock type to use. Only required if TbYield is true.
                            boost::fibers::condition_variable,  // The condition variable type to use. Only required if TbYield is true.
                            false>;                             // If the threads should yield.
                        FiberPool pool(uiNumThreadsInBlock[0], uiNumThreadsInBlock[0]);

                        // Execute the blocks serially.
                        for(this->AccFibers::m_v3uiGridBlockIdx[2] = 0; this->AccFibers::m_v3uiGridBlockIdx[2]<v3uiGridBlockExtents[2]; ++this->AccFibers::m_v3uiGridBlockIdx[2])
                        {
                            for(this->AccFibers::m_v3uiGridBlockIdx[1] = 0; this->AccFibers::m_v3uiGridBlockIdx[1]<v3uiGridBlockExtents[1]; ++this->AccFibers::m_v3uiGridBlockIdx[1])
                            {
                                for(this->AccFibers::m_v3uiGridBlockIdx[0] = 0; this->AccFibers::m_v3uiGridBlockIdx[0]<v3uiGridBlockExtents[0]; ++this->AccFibers::m_v3uiGridBlockIdx[0])
                                {
                                    // Execute the block thread in parallel using cooperative multi-threading.
                                    Vec3<> v3uiBlockThreadIdx(Vec3<>::zeros());
                                    for(v3uiBlockThreadIdx[2] = 0; v3uiBlockThreadIdx[2]<v3uiBlockThreadExtents[2]; ++v3uiBlockThreadIdx[2])
                                    {
                                        for(v3uiBlockThreadIdx[1] = 0; v3uiBlockThreadIdx[1]<v3uiBlockThreadExtents[1]; ++v3uiBlockThreadIdx[1])
                                        {
                                            for(v3uiBlockThreadIdx[0] = 0; v3uiBlockThreadIdx[0]<v3uiBlockThreadExtents[0]; ++v3uiBlockThreadIdx[0])
                                            {
                                                // The v3uiBlockThreadIdx is required to be copied in from the environment because if the fiber is immediately suspended the variable is already changed for the next iteration/thread.
                                                auto fiberKernelFct =
                                                    [&, v3uiBlockThreadIdx]()
                                                    {
                                                        fiberKernel(
                                                            v3uiBlockThreadIdx,
                                                            std::forward<TKernelFunctor>(kernelFunctor),
                                                            std::forward<TArgs>(args)...);
                                                    };
                                                m_vFuturesInBlock.emplace_back(
                                                    pool.enqueueTask(
                                                        fiberKernelFct));
                                            }
                                        }
                                    }

                                    // Wait for the completion of the kernels.
                                    std::for_each(m_vFuturesInBlock.begin(), m_vFuturesInBlock.end(),
                                        [](boost::fibers::future<void> & t)
                                        {
                                            t.wait();
                                        }
                                    );
                                    // Clean up.
                                    m_vFuturesInBlock.clear();

                                    this->AccFibers::m_mFibersToIndices.clear();
                                    this->AccFibers::m_mFibersToBarrier.clear();

                                    // After a block has been processed, the shared memory can be deleted.
                                    this->AccFibers::m_vvuiSharedMem.clear();
                                }
                            }
                        }
                        // After all blocks have been processed, the external shared memory can be deleted.
                        this->AccFibers::m_vuiExternalSharedMem.reset();
                    }
                private:
                    //-----------------------------------------------------------------------------
                    //! The fiber entry point.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto fiberKernel(
                        Vec3<> const & v3uiBlockThreadIdx,
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                        auto const idFiber(boost::this_fiber::get_id());

                        // Set the master thread id.
                        if(v3uiBlockThreadIdx[0] == 0 && v3uiBlockThreadIdx[1] == 0 && v3uiBlockThreadIdx[2] == 0)
                        {
                            this->AccFibers::m_idMasterFiber = idFiber;
                        }

                        // We can not use the default syncBlockThreads here because it searches inside m_mFibersToBarrier for the thread id.
                        // Concurrently searching while others use emplace is unsafe!
                        std::map<boost::fibers::fiber::id, UInt>::iterator itFiberToBarrier;

                        // Save the fiber id, and index.
                        this->AccFibers::m_mFibersToIndices.emplace(idFiber, v3uiBlockThreadIdx);
                        itFiberToBarrier = this->AccFibers::m_mFibersToBarrier.emplace(idFiber, 0).first;

                        // Sync all threads so that the maps with thread id's are complete and not changed after here.
                        this->AccFibers::syncBlockThreads(itFiberToBarrier);

                        // Execute the kernel itself.
                        kernelFunctor(
                            (*static_cast<AccFibers const *>(this)),
                            std::forward<TArgs>(args)...);

                        // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recycled (then duplicate) fiber id!
                        this->AccFibers::syncBlockThreads(itFiberToBarrier);
                    }

                public:
                    devs::cpu::detail::StreamCpu m_Stream;

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
            //! The fibers accelerator executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::fibers::detail::ExecFibers>
            {
                using type = accs::fibers::detail::AccFibers;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The fibers accelerator executor event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::fibers::detail::ExecFibers>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The fibers accelerator executor executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::fibers::detail::ExecFibers>
            {
                using type = accs::fibers::detail::ExecFibers;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The fibers accelerator executor device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::fibers::detail::ExecFibers>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The fibers accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::fibers::detail::ExecFibers>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The fibers accelerator executor stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::fibers::detail::ExecFibers>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
            //#############################################################################
            //! The fibers accelerator executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::fibers::detail::ExecFibers>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::fibers::detail::ExecFibers const & exec)
                -> devs::cpu::detail::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
