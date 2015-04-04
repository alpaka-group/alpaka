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

// Base classes.
#include <alpaka/accs/fibers/WorkDiv.hpp>           // WorkDivFibers
#include <alpaka/accs/fibers/Idx.hpp>               // IdxFibers
#include <alpaka/accs/fibers/Atomic.hpp>            // AtomicFibers
#include <alpaka/accs/fibers/Barrier.hpp>           // BarrierFibers

// User functionality.
#include <alpaka/host/Mem.hpp>                      // Copy
#include <alpaka/host/mem/Space.hpp>                // SpaceHost
#include <alpaka/accs/fibers/Stream.hpp>            // StreamFibers
#include <alpaka/accs/fibers/Event.hpp>             // EventFibers
#include <alpaka/accs/fibers/Dev.hpp>               // Devices

// Specialized traits.
#include <alpaka/traits/Acc.hpp>                    // AccType
#include <alpaka/traits/Exec.hpp>                   // ExecType
#include <alpaka/traits/Mem.hpp>                    // SpaceType

// Implementation details.
#include <alpaka/accs/fibers/Common.hpp>
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>       // ConcurrentExecPool

#include <boost/predef.h>                           // workarounds

#include <cassert>                                  // assert
#include <stdexcept>                                // std::except

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The fibers accelerator.
        //-----------------------------------------------------------------------------
        namespace fibers
        {
            //-----------------------------------------------------------------------------
            //! The fibers accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                class KernelExecFibers;

                //#############################################################################
                //! The fibers accelerator.
                //!
                //! This accelerator allows parallel kernel execution on the host.
                //! It uses boost::fibers to implement the cooperative parallelism.
                //! By using fibers the shared memory can reside in the closest memory/cache available.
                //! Furthermore there is no false sharing between neighboring threads as it is the case in real multi-threading.
                //#############################################################################
                class AccFibers :
                    protected WorkDivFibers,
                    protected IdxFibers,
                    protected AtomicFibers
                {
                public:
                    using MemSpace = mem::SpaceHost;

                    friend class ::alpaka::accs::fibers::detail::KernelExecFibers;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccFibers(
                        TWorkDiv const & workDiv) :
                            WorkDivFibers(workDiv),
                            IdxFibers(m_mFibersToIndices, m_v3uiGridBlockIdx),
                            AtomicFibers(),
                            m_v3uiGridBlockIdx(0u),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads, dim::Dim1>(workDiv)[0u])
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccFibers(AccFibers const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccFibers(AccFibers &&) = delete;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccFibers const &) -> AccFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccFibers() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The requested indices.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                    -> DimToVecT<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit, TDim>(
                            *static_cast<IdxFibers const *>(this),
                            *static_cast<WorkDivFibers const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                    -> DimToVecT<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                            *static_cast<WorkDivFibers const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Execute the atomic operation on the given address with the given value.
                    //! \return The old value before executing the atomic operation.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOp,
                        typename T>
                    ALPAKA_FCT_ACC auto atomicOp(
                        T * const addr,
                        T const & value) const
                    -> T
                    {
                        return atomic::atomicOp<TOp, T>(
                            addr,
                            value,
                            *static_cast<AtomicFibers const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads() const
                    -> void
                    {
                        auto const idFiber(boost::this_fiber::get_id());
                        auto const itFind(m_mFibersToBarrier.find(idFiber));

                        syncBlockThreads(itFind);
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads(
                        std::map<boost::fibers::fiber::id, UInt>::iterator const & itFind) const
                    -> void
                    {
                        assert(itFind != m_mFibersToBarrier.end());

                        auto & uiBarrierIdx(itFind->second);
                        std::size_t const uiModBarrierIdx(uiBarrierIdx % 2);

                        auto & bar(m_abarSyncFibers[uiModBarrierIdx]);

                        // (Re)initialize a barrier if this is the first fiber to reach it.
                        if(bar.getNumFibersToWaitFor() == 0)
                        {
                            // No DCLP required because there can not be an interruption in between the check and the reset.
                            bar.reset(m_uiNumThreadsPerBlock);
                        }

                        // Wait for the barrier.
                        bar.wait();
                        ++uiBarrierIdx;
                    }
                public:
                    //-----------------------------------------------------------------------------
                    //! \return Allocates block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T,
                        UInt TuiNumElements>
                    ALPAKA_FCT_ACC_NO_CUDA auto allocBlockSharedMem() const
                    -> T *
                    {
                        static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                        // Assure that all fibers have executed the return of the last allocBlockSharedMem function (if there was one before).
                        syncBlockThreads();

                        // Arbitrary decision: The fiber that was created first has to allocate the memory.
                        if(m_idMasterFiber == boost::this_fiber::get_id())
                        {
                            // \TODO: C++14 std::make_unique would be better.
                            m_vvuiSharedMem.emplace_back(
                                std::unique_ptr<uint8_t[]>(
                                    reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                        }
                        syncBlockThreads();

                        return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The pointer to the externally allocated block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T>
                    ALPAKA_FCT_ACC_NO_CUDA auto getBlockSharedExternMem() const
                    -> T *
                    {
                        return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                    }

#ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
                protected:
#else
                private:
#endif
                    // getXxxIdx
                    FiberIdToIdxMap mutable m_mFibersToIndices;                 //!< The mapping of fibers id's to fibers indices.
                    Vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                    // syncBlockThreads
                    UInt const m_uiNumThreadsPerBlock;                            //!< The number of threads per block the barrier has to wait for.
                    std::map<
                        boost::fibers::fiber::id,
                        UInt> mutable m_mFibersToBarrier;                       //!< The mapping of fibers id's to their current barrier.
                    FiberBarrier mutable m_abarSyncFibers[2];                   //!< The barriers for the synchronization of fibers.
                    //!< We have the keep to current and the last barrier because one of the fibers can reach the next barrier before another fiber was wakeup from the last one and has checked if it can run.

                    // allocBlockSharedMem
                    boost::fibers::fiber::id mutable m_idMasterFiber;           //!< The id of the master fiber.
                    std::vector<
                        std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };

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
                class KernelExecFibers :
                    private AccFibers
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST KernelExecFibers(
                        TWorkDiv const & workDiv,
                        StreamFibers const &):
                            AccFibers(workDiv),
                            m_vFuturesInBlock()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST KernelExecFibers(
                        KernelExecFibers const & other):
                            AccFibers(static_cast<WorkDivFibers const &>(other)),
                            m_vFuturesInBlock()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST KernelExecFibers(
                        KernelExecFibers && other) :
                            AccFibers(static_cast<WorkDivFibers &&>(other))
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(KernelExecFibers const &) -> KernelExecFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~KernelExecFibers() = default;
#else
                    ALPAKA_FCT_HOST virtual ~KernelExecFibers() noexcept = default;
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

                        Vec<3u> const v3uiGridBlockExtents(this->AccFibers::getWorkDiv<Grid, Blocks, dim::Dim3>());
                        Vec<3u> const v3uiBlockThreadExtents(this->AccFibers::getWorkDiv<Block, Threads, dim::Dim3>());

                        auto const uiBlockSharedExternMemSizeBytes(getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccFibers>(
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
                                    Vec<3u> v3uiBlockThreadIdx(0u);
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

                                    // Join all the threads.
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
                        Vec<3u> const & v3uiBlockThreadIdx,
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

                private:
                    std::vector<boost::fibers::future<void>> mutable m_vFuturesInBlock; //!< The futures of the fibers in the current block.
                };
            }
        }
    }

    using AccFibers = accs::fibers::detail::AccFibers;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The fibers accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::fibers::detail::KernelExecFibers>
            {
                using type = AccFibers;
            };

            //#############################################################################
            //! The fibers accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::fibers::detail::AccFibers>
            {
                using type = accs::fibers::detail::AccFibers;
            };

            //#############################################################################
            //! The fibers accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::fibers::detail::AccFibers>
            {
                static auto getAccName()
                -> std::string
                {
                    return "AccFibers";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The fibers accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::fibers::detail::AccFibers>
            {
                using type = accs::fibers::detail::EventFibers;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The fibers accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::fibers::detail::AccFibers>
            {
                using type = accs::fibers::detail::KernelExecFibers;
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The fibers accelerator memory space trait specialization.
            //#############################################################################
            template<>
            struct SpaceType<
                accs::fibers::detail::AccFibers>
            {
                using type = alpaka::mem::SpaceHost;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The fibers accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::fibers::detail::AccFibers>
            {
                using type = accs::fibers::detail::StreamFibers;
            };

            //#############################################################################
            //! The fibers accelerator kernel executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::fibers::detail::KernelExecFibers>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::fibers::detail::KernelExecFibers const &)
                -> accs::fibers::detail::StreamFibers
                {
                    return accs::fibers::detail::StreamFibers(
                        accs::fibers::detail::DevManFibers::getDevByIdx(0));
                }
            };
        }
    }
}
