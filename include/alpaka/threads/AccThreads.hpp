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

//#define ALPAKA_THREADS_NO_POOL    // Define this to recreate all of the threads between executing blocks.
                                    // NOTE: Using the thread pool should be massively faster in nearly every case!

#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
    #define ALPAKA_THREADS_NO_POOL
#endif

// Base classes.
#include <alpaka/threads/AccThreadsFwd.hpp>
#include <alpaka/threads/WorkDiv.hpp>               // WorkDivThreads
#include <alpaka/threads/Idx.hpp>                   // IdxThreads
#include <alpaka/threads/Atomic.hpp>                // AtomicThreads
#include <alpaka/threads/Barrier.hpp>               // BarrierThreads

// User functionality.
#include <alpaka/host/Mem.hpp>                      // MemCopy
#include <alpaka/threads/Stream.hpp>                // StreamThreads
#include <alpaka/threads/Event.hpp>                 // EventThreads
#include <alpaka/threads/Device.hpp>                // Devices

// Specialized traits.
#include <alpaka/core/KernelExecCreator.hpp>        // KernelExecCreator

// Implementation details.
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>
#ifndef ALPAKA_THREADS_NO_POOL
    #include <alpaka/core/ConcurrentExecPool.hpp>  // ConcurrentExecPool
#endif

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply
#include <boost/predef.h>                           // workarounds

#include <vector>                                   // std::vector
#include <thread>                                   // std::thread
#include <map>                                      // std::map
#include <algorithm>                                // std::for_each
#include <array>                                    // std::array
#include <cassert>                                  // assert
#include <stdexcept>                                // std::runtime_error
#include <string>                                   // std::to_string

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            template<
                typename TAcceleratedKernel>
            class KernelExecutorThreads;

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
                using MemSpace = mem::MemSpaceHost;

                template<
                    typename TAcceleratedKernel>
                friend class KernelExecutorThreads;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccThreads() :
                    WorkDivThreads(),
                    IdxThreads(m_mThreadsToIndices, m_v3uiGridBlockIdx),
                    AtomicThreads()
                {}
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
                ALPAKA_FCT_ACC_NO_CUDA AccThreads & operator=(AccThreads const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL     // threads/AccThreads.hpp(134): error : the declared exception specification is incompatible with the generated one
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AccThreads() = default;
#else
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AccThreads() noexcept = default;
#endif

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested indices.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin, 
                    typename TUnit, 
                    typename TDim = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA DimToVecT<TDim> getIdx() const
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
                ALPAKA_FCT_ACC_NO_CUDA DimToVecT<TDim> getWorkDiv() const
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
                ALPAKA_FCT_ACC T atomicOp(
                    T * const addr,
                    T const & value) const
                {
                    return atomic::atomicOp<TOp, T>(
                        addr,
                        value,
                        *static_cast<AtomicThreads const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockKernels() const
                {
                    auto const idThread(std::this_thread::get_id());
                    auto const itFind(m_mThreadsToBarrier.find(idThread));

                    syncBlockKernels(itFind);
                }
            private:
                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockKernels(
                    std::map<std::thread::id, UInt>::iterator const & itFind) const
                {
                    assert(itFind != m_mThreadsToBarrier.end());

                    auto & uiBarIdx(itFind->second);
                    UInt const uiBarrierIdx(uiBarIdx % 2);

                    auto & bar(m_abarSyncThreads[uiBarrierIdx]);

                    // (Re)initialize a barrier if this is the first thread to reach it.
                    if(bar.getNumThreadsToWaitFor() == 0)
                    {
                        std::lock_guard<std::mutex> lock(m_mtxBarrier);
                        if(bar.getNumThreadsToWaitFor() == 0)
                        {
                            bar.reset(m_uiNumKernelsPerBlock);
                        }
                    }

                    // Wait for the barrier.
                    bar.wait();
                    ++uiBarIdx;
                }
            protected:
                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T, 
                    UInt TuiNumElements>
                ALPAKA_FCT_ACC_NO_CUDA T * allocBlockSharedMem() const
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    // Assure that all threads have executed the return of the last allocBlockSharedMem function (if there was one before).
                    syncBlockKernels();

                    // Arbitrary decision: The thread that was created first has to allocate the memory.
                    if(m_idMasterThread == std::this_thread::get_id())
                    {
                        // \TODO: C++14 std::make_unique would be better.
                        m_vvuiSharedMem.emplace_back(
                            std::unique_ptr<uint8_t[]>(
                                reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                    }
                    syncBlockKernels();

                    return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T>
                ALPAKA_FCT_ACC_NO_CUDA T * getBlockSharedExternMem() const
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

                // syncBlockKernels
                UInt mutable m_uiNumKernelsPerBlock;                        //!< The number of kernels per block the barrier has to wait for.
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
                static void yield()
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
            template<
                typename TAcceleratedKernel>
            class KernelExecutorThreads :
                private TAcceleratedKernel,
                private IAcc<AccThreads>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv, 
                    typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutorThreads(
                    TWorkDiv const & workDiv, 
                    StreamThreads const &,
                    TKernelConstrArgs && ... args) :
                        TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...),
                        IAcc<AccThreads>(),
#ifdef ALPAKA_THREADS_NO_POOL
                        m_vThreadsInBlock(),
#else
                        m_vFuturesInBlock(),
#endif
                        m_mtxMapInsert()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    (*static_cast<WorkDivThreads *>(this)) = workDiv;

                    this->AccThreads::m_uiNumKernelsPerBlock = workdiv::getWorkDiv<Block, Kernels, dim::Dim1>(workDiv)[0];
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorThreads(
                    KernelExecutorThreads const & other) :
                        TAcceleratedKernel(other),
                        IAcc<AccThreads>(),
#ifdef ALPAKA_THREADS_NO_POOL
                        m_vThreadsInBlock(),
#else
                        m_vFuturesInBlock(),
#endif
                        m_mtxMapInsert()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    (*static_cast<WorkDivThreads *>(this)) = (*static_cast<WorkDivThreads const *>(&other));

                    this->AccThreads::m_uiNumKernelsPerBlock = other.getBlockKernelsExtents().prod();
                }
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorThreads(
                    KernelExecutorThreads && other) :
                        TAcceleratedKernel(std::move(other)),
                        IAcc<AccThreads>(),
#ifdef ALPAKA_THREADS_NO_POOL
                        m_vThreadsInBlock(),
#else
                        m_vFuturesInBlock(),
#endif
                        m_mtxMapInsert()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    (*static_cast<WorkDivThreads *>(this)) = (*static_cast<WorkDivThreads const *>(&other));

                    this->AccThreads::m_uiNumKernelsPerBlock = other.getBlockKernelsExtents().prod();
                }
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorThreads & operator=(KernelExecutorThreads const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                ALPAKA_FCT_HOST virtual ~KernelExecutorThreads() = default;
#else
                ALPAKA_FCT_HOST virtual ~KernelExecutorThreads() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<
                    typename... TArgs>
                ALPAKA_FCT_HOST void operator()(
                    TArgs && ... args) const
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    Vec<3u> const v3uiGridBlocksExtents(this->AccThreads::getWorkDiv<Grid, Blocks, dim::Dim3>());
                    Vec<3u> const v3uiBlockKernelsExtents(this->AccThreads::getWorkDiv<Block, Kernels, dim::Dim3>());

                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(v3uiBlockKernelsExtents, std::forward<TArgs>(args)...));
                    this->AccThreads::m_vuiExternalSharedMem.reset(
                        new uint8_t[uiBlockSharedExternMemSizeBytes]);

#ifndef ALPAKA_THREADS_NO_POOL
                    auto const uiNumKernelsInBlock(this->AccThreads::getWorkDiv<Block, Kernels, dim::Dim1>());
                    // When using the thread pool the threads are yielding because this is faster. 
                    // Using condition variables and going to sleep is very costly for real threads. 
                    // Especially when the time to wait is really short (syncBlockKernels) yielding is much faster.
                    using ThreadPool = alpaka::detail::ConcurrentExecPool<
                        std::thread,                // The concurrent execution type.
                        std::promise,               // The promise type.
                        ThreadPoolCurrentException, // The type returning the current exception.
                        ThreadPoolYield>;           // The type yielding the current concurrent execution.
                    ThreadPool pool(uiNumKernelsInBlock[0], uiNumKernelsInBlock[0]);
#endif
                    // Execute the blocks serially.
                    for(UInt bz(0); bz<v3uiGridBlocksExtents[2]; ++bz)
                    {
                        this->AccThreads::m_v3uiGridBlockIdx[2] = bz;
                        for(UInt by(0); by<v3uiGridBlocksExtents[1]; ++by)
                        {
                            this->AccThreads::m_v3uiGridBlockIdx[1] = by;
                            for(UInt bx(0); bx<v3uiGridBlocksExtents[0]; ++bx)
                            {
                                this->AccThreads::m_v3uiGridBlockIdx[0] = bx;

                                // Execute the kernels in parallel threads.
                                Vec<3u> v3uiBlockKernelIdx;
                                for(UInt tz(0); tz<v3uiBlockKernelsExtents[2]; ++tz)
                                {
                                    v3uiBlockKernelIdx[2] = tz;
                                    for(UInt ty(0); ty<v3uiBlockKernelsExtents[1]; ++ty)
                                    {
                                        v3uiBlockKernelIdx[1] = ty;
                                        for(UInt tx(0); tx<v3uiBlockKernelsExtents[0]; ++tx)
                                        {
                                            v3uiBlockKernelIdx[0] = tx;

                                            // Create a thread.
                                            // The v3uiBlockKernelIdx is required to be copied in from the environment because if the thread is immediately suspended the variable is already changed for the next iteration/thread.

#if BOOST_COMP_MSVC     // VC is missing the this pointer type in the signature of &KernelExecutorThreads::threadKernel<TArgs...>
                                            auto threadKernelFct = 
                                                [&, v3uiBlockKernelIdx]()
                                                {
                                                    threadKernel<TArgs...>(v3uiBlockKernelIdx, std::forward<TArgs>(args)...); 
                                                };
    #ifdef ALPAKA_THREADS_NO_POOL
                                            m_vThreadsInBlock.push_back(std::thread(threadKernelFct));
    #else
                                            m_vFuturesInBlock.emplace_back(pool.enqueueTask(threadKernelFct));
    #endif
#elif BOOST_COMP_GNUC   // But GCC < 4.9.0 can not compile variadic templates inside lambdas correctly if the variadic argument is not in the parameter list.
                                            auto threadKernelFct(
                                                [this](Vec<3u> const v3uiBlockKernelIdx, TArgs ... args)
                                                {
                                                    threadKernel<TArgs...>(v3uiBlockKernelIdx, std::forward<TArgs>(args)...); 
                                                });
    #ifdef ALPAKA_THREADS_NO_POOL
                                            m_vThreadsInBlock.push_back(std::thread(threadKernelFct, v3uiBlockKernelIdx, args...));
    #else
                                            m_vFuturesInBlock.emplace_back(pool.enqueueTask(threadKernelFct, v3uiBlockKernelIdx, args...));
    #endif
#else
    #ifdef ALPAKA_THREADS_NO_POOL
                                            m_vThreadsInBlock.push_back(std::thread(&KernelExecutorThreads::threadKernel<TArgs...>, this, v3uiBlockKernelIdx, std::forward<TArgs>(args)...));
    #else
                                            m_vFuturesInBlock.emplace_back(pool.enqueueTask(&KernelExecutorThreads::threadKernel<TArgs...>, this, v3uiBlockKernelIdx, std::forward<TArgs>(args)...));
    #endif
#endif
                                        }
                                    }
                                }
#ifdef ALPAKA_THREADS_NO_POOL
                                // Join all the threads.
                                std::for_each(m_vThreadsInBlock.begin(), m_vThreadsInBlock.end(),
                                    [](std::thread & t)
                                    {
                                        t.join();
                                    }
                                );
                                // Clean up.
                                m_vThreadsInBlock.clear();
#else
                                // Join all the threads.
                                std::for_each(m_vFuturesInBlock.begin(), m_vFuturesInBlock.end(),
                                    [](std::future<void> & t)
                                    {
                                        t.wait();
                                    }
                                );
                                // Clean up.
                                m_vFuturesInBlock.clear();
#endif
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
                    typename... TArgs>
                ALPAKA_FCT_HOST void threadKernel(
                    Vec<3u> const & v3uiBlockKernelIdx, 
                    TArgs && ... args) const
                {
                    // We have to store the thread data before the kernel is calling any of the methods of this class depending on them.
                    auto const idThread(std::this_thread::get_id());

                    // Set the master thread id.
                    if(v3uiBlockKernelIdx[0] == 0 && v3uiBlockKernelIdx[1] == 0 && v3uiBlockKernelIdx[2] == 0)
                    {
                        this->AccThreads::m_idMasterThread = idThread;
                    }

                    // We can not use the default syncBlockKernels here because it searches inside m_mThreadsToBarrier for the thread id. 
                    // Concurrently searching while others use emplace is unsafe!
                    std::map<std::thread::id, UInt>::iterator itThreadToBarrier;

                    {
                        // The insertion of elements has to be done one thread at a time.
                        std::lock_guard<std::mutex> lock(m_mtxMapInsert);

                        // Save the thread id, and index.
                        this->AccThreads::m_mThreadsToIndices.emplace(idThread, v3uiBlockKernelIdx);
                        itThreadToBarrier = this->AccThreads::m_mThreadsToBarrier.emplace(idThread, 0).first;
                    }

                    // Sync all fibers so that the maps with fiber id's are complete and not changed after here.
                    this->AccThreads::syncBlockKernels(itThreadToBarrier);

                    // Execute the kernel itself.
                    this->TAcceleratedKernel::operator()(
                        (*static_cast<IAcc<AccThreads> const *>(this)),
                        std::forward<TArgs>(args)...);

                    // We have to sync all threads here because if a thread would finish before all threads have been started, the new thread could get a recycled (then duplicate) thread id!
                    this->AccThreads::syncBlockKernels(itThreadToBarrier);
                }

            private:
#ifdef ALPAKA_THREADS_NO_POOL
                std::vector<std::thread> mutable m_vThreadsInBlock;       //!< The threads executing the current block.
#else
                std::vector<std::future<void>> mutable m_vFuturesInBlock; //!< The futures of the threads in the current block.
#endif
                std::mutex mutable m_mtxMapInsert;

                Vec<3u> m_v3uiGridBlocksExtents;
                Vec<3u> m_v3uiBlockKernelsExtents;
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The threads accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename AcceleratedKernel>
            struct AccType<
                threads::detail::KernelExecutorThreads<AcceleratedKernel>>
            {
                using type = AccThreads;
            };
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The threads accelerator kernel executor builder.
        //#############################################################################
        template<
            typename TKernel, 
            typename... TKernelConstrArgs>
        class KernelExecCreator<
            AccThreads, 
            TKernel, 
            TKernelConstrArgs...>
        {
        public:
            using AcceleratedKernel = typename boost::mpl::apply<TKernel, AccThreads>::type;
            using AcceleratedKernelExecutorExtent = KernelExecutorExtent<threads::detail::KernelExecutorThreads<AcceleratedKernel>, TKernelConstrArgs...>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST AcceleratedKernelExecutorExtent operator()(
                TKernelConstrArgs && ... args) const
            {
                return AcceleratedKernelExecutorExtent(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
