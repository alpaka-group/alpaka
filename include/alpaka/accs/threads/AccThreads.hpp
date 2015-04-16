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
#include <alpaka/accs/threads/WorkDiv.hpp>          // WorkDivThreads
#include <alpaka/accs/threads/Idx.hpp>              // IdxThreads
#include <alpaka/accs/threads/Atomic.hpp>           // AtomicThreads
#include <alpaka/accs/threads/Barrier.hpp>          // BarrierThreads

// User functionality.
#include <alpaka/host/Mem.hpp>                      // Copy
#include <alpaka/host/mem/Space.hpp>                // SpaceHost
#include <alpaka/accs/threads/Stream.hpp>           // StreamThreads
#include <alpaka/accs/threads/Event.hpp>            // EventThreads
#include <alpaka/accs/threads/Dev.hpp>              // Devices

// Specialized traits.
#include <alpaka/traits/Acc.hpp>                    // AccType
#include <alpaka/traits/Exec.hpp>                   // ExecType
#include <alpaka/traits/Mem.hpp>                    // SpaceType

// Implementation details.
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>       // ConcurrentExecPool

#include <boost/predef.h>                           // workarounds

#include <vector>                                   // std::vector
#include <thread>                                   // std::thread
#include <map>                                      // std::map
#include <algorithm>                                // std::for_each
#include <array>                                    // std::array
#include <cassert>                                  // assert
#include <string>                                   // std::to_string

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The threads accelerator.
        //-----------------------------------------------------------------------------
        namespace threads
        {
            //-----------------------------------------------------------------------------
            //! The threads accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                class KernelExecThreads;

                //#############################################################################
                //! The threads accelerator.
                //!
                //! This accelerator allows parallel kernel execution on the host.
                //! It uses C++11 std::threads to implement the parallelism.
                //#############################################################################
                class AccThreads :
                    protected WorkDivThreads,
                    protected IdxThreads,
                    protected AtomicThreads
                {
                public:
                    using MemSpace = mem::SpaceHost;

                    friend class ::alpaka::accs::threads::detail::KernelExecThreads;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccThreads(
                        TWorkDiv const & workDiv) :
                            WorkDivThreads(workDiv),
                            IdxThreads(m_mThreadsToIndices, m_v3uiGridBlockIdx),
                            AtomicThreads(),
                            m_v3uiGridBlockIdx(Vec<3u>::zeros()),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads, dim::Dim1>(workDiv)[0u])
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccThreads(AccThreads const &) = delete;
    #if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccThreads(AccThreads &&) = delete;
    #endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccThreads const &) -> AccThreads & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL     // threads/AccThreads.hpp(134): error : the declared exception specification is incompatible with the generated one
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccThreads() = default;
    #else
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccThreads() noexcept = default;
    #endif

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
                            *static_cast<IdxThreads const *>(this),
                            *static_cast<WorkDivThreads const *>(this));
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
                            *static_cast<WorkDivThreads const *>(this));
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
                            *static_cast<AtomicThreads const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads() const
                    -> void
                    {
                        auto const idThread(std::this_thread::get_id());
                        auto const itFind(m_mThreadsToBarrier.find(idThread));

                        syncBlockThreads(itFind);
                    }
                private:
                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads(
                        std::map<std::thread::id, UInt>::iterator const & itFind) const
                    -> void
                    {
                        assert(itFind != m_mThreadsToBarrier.end());

                        auto & uiBarrierIdx(itFind->second);
                        std::size_t const uiModBarrierIdx(uiBarrierIdx % 2);

                        auto & bar(m_abarSyncThreads[uiModBarrierIdx]);

                        // (Re)initialize a barrier if this is the first thread to reach it.
                        if(bar.getNumThreadsToWaitFor() == 0)
                        {
                            std::lock_guard<std::mutex> lock(m_mtxBarrier);
                            if(bar.getNumThreadsToWaitFor() == 0)
                            {
                                bar.reset(m_uiNumThreadsPerBlock);
                            }
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

                        // Assure that all threads have executed the return of the last allocBlockSharedMem function (if there was one before).
                        syncBlockThreads();

                        // Arbitrary decision: The thread that was created first has to allocate the memory.
                        if(m_idMasterThread == std::this_thread::get_id())
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
                    // getIdx
                    detail::ThreadIdToIdxMap mutable m_mThreadsToIndices;       //!< The mapping of thread id's to thread indices.
                    Vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                    // syncBlockThreads
                    UInt const m_uiNumThreadsPerBlock;                          //!< The number of threads per block the barrier has to wait for.
                    std::map<
                        std::thread::id,
                        UInt> mutable m_mThreadsToBarrier;                      //!< The mapping of thread id's to their current barrier.
                    std::mutex mutable m_mtxBarrier;
                    detail::ThreadBarrier mutable m_abarSyncThreads[2];         //!< The barriers for the synchronization of threads.
                    //!< We have to keep the current and the last barrier because one of the threads can reach the next barrier before a other thread was wakeup from the last one and has checked if it can run.

                    // allocBlockSharedMem
                    std::thread::id mutable m_idMasterThread;                   //!< The id of the master thread.
                    std::vector<
                        std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };

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
                //! The type given to the ConcurrentExecPool for returning the current exception.
                //#############################################################################
                struct ThreadPoolCurrentException
                {
                    //-----------------------------------------------------------------------------
                    //! \return The current exception.
                    //-----------------------------------------------------------------------------
                    static auto current_exception()
                    -> std::result_of<decltype(&std::current_exception)()>::type
                    {
                        return std::current_exception();
                    }
                };

                //#############################################################################
                //! The threads accelerator executor.
                //#############################################################################
                class KernelExecThreads :
                    private AccThreads
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST KernelExecThreads(
                        TWorkDiv const & workDiv,
                        StreamThreads const &) :
                            AccThreads(workDiv),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST KernelExecThreads(
                        KernelExecThreads const & other) :
                            AccThreads(static_cast<WorkDivThreads const &>(other)),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST KernelExecThreads(
                        KernelExecThreads && other) :
                            AccThreads(static_cast<WorkDivThreads &&>(other)),
                            m_vFuturesInBlock(),
                            m_mtxMapInsert()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(KernelExecThreads const &) -> KernelExecThreads & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~KernelExecThreads() = default;
    #else
                    ALPAKA_FCT_HOST virtual ~KernelExecThreads() noexcept = default;
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

                        Vec<3u> const v3uiGridBlockExtents(this->AccThreads::getWorkDiv<Grid, Blocks, dim::Dim3>());
                        Vec<3u> const v3uiBlockThreadExtents(this->AccThreads::getWorkDiv<Block, Threads, dim::Dim3>());

                        auto const uiBlockSharedExternMemSizeBytes(getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccThreads>(
                            v3uiBlockThreadExtents,
                            std::forward<TArgs>(args)...));
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
    #endif
                        this->AccThreads::m_vuiExternalSharedMem.reset(
                            new uint8_t[uiBlockSharedExternMemSizeBytes]);

                        auto const uiNumThreadsInBlock(this->AccThreads::getWorkDiv<Block, Threads, dim::Dim1>());
                        // When using the thread pool the threads are yielding because this is faster.
                        // Using condition variables and going to sleep is very costly for real threads.
                        // Especially when the time to wait is really short (syncBlockThreads) yielding is much faster.
                        using ThreadPool = alpaka::detail::ConcurrentExecPool<
                            std::thread,                // The concurrent execution type.
                            std::promise,               // The promise type.
                            ThreadPoolCurrentException, // The type returning the current exception.
                            ThreadPoolYield>;           // The type yielding the current concurrent execution.
                        ThreadPool pool(uiNumThreadsInBlock[0], uiNumThreadsInBlock[0]);

                        // Execute the blocks serially.
                        for(this->AccThreads::m_v3uiGridBlockIdx[2] = 0; this->AccThreads::m_v3uiGridBlockIdx[2]<v3uiGridBlockExtents[2]; ++this->AccThreads::m_v3uiGridBlockIdx[2])
                        {
                            for(this->AccThreads::m_v3uiGridBlockIdx[1] = 0; this->AccThreads::m_v3uiGridBlockIdx[1]<v3uiGridBlockExtents[1]; ++this->AccThreads::m_v3uiGridBlockIdx[1])
                            {
                                for(this->AccThreads::m_v3uiGridBlockIdx[0] = 0; this->AccThreads::m_v3uiGridBlockIdx[0]<v3uiGridBlockExtents[0]; ++this->AccThreads::m_v3uiGridBlockIdx[0])
                                {
                                    // Execute the threads in parallel.
                                    Vec<3u> v3uiBlockThreadIdx(Vec<3u>::zeros());
                                    for(v3uiBlockThreadIdx[2] = 0; v3uiBlockThreadIdx[2]<v3uiBlockThreadExtents[2]; ++v3uiBlockThreadIdx[2])
                                    {
                                        for(v3uiBlockThreadIdx[1] = 0; v3uiBlockThreadIdx[1]<v3uiBlockThreadExtents[1]; ++v3uiBlockThreadIdx[1])
                                        {
                                            for(v3uiBlockThreadIdx[0] = 0; v3uiBlockThreadIdx[0]<v3uiBlockThreadExtents[0]; ++v3uiBlockThreadIdx[0])
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

                                    // After a block has been processed, the shared memory can be deleted.
                                    this->AccThreads::m_vvuiSharedMem.clear();
                                }
                            }
                        }
                        // After all blocks have been processed, the external shared memory can be deleted.
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
                        Vec<3u> const & v3uiBlockThreadIdx,
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

                private:
                    std::vector<std::future<void>> mutable m_vFuturesInBlock; //!< The futures of the threads in the current block.

                    std::mutex mutable m_mtxMapInsert;
                };
            }
        }
    }

    using AccThreads = accs::threads::detail::AccThreads;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The threads accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::threads::detail::KernelExecThreads>
            {
                using type = AccThreads;
            };

            //#############################################################################
            //! The threads accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::AccThreads;
            };

            //#############################################################################
            //! The threads accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::threads::detail::AccThreads>
            {
                static auto getAccName()
                -> std::string
                {
                    return "AccThreads";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The threads accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::EventThreads;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The threads accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::KernelExecThreads;
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The threads accelerator memory space trait specialization.
            //#############################################################################
            template<>
            struct SpaceType<
                accs::threads::detail::AccThreads>
            {
                using type = alpaka::mem::SpaceHost;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The threads accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::StreamThreads;
            };

            //#############################################################################
            //! The threads accelerator kernel executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::threads::detail::KernelExecThreads>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::threads::detail::KernelExecThreads const &)
                -> accs::threads::detail::StreamThreads
                {
                    return accs::threads::detail::StreamThreads(
                        accs::threads::detail::DevManThreads::getDevByIdx(0));
                }
            };
        }
    }
}
