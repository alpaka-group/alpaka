/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

//#define ALPAKA_THREADS_NO_POOL    // Define this to recreate all of the threads between executing blocks.
                                    // NOTE: Using the thread pool should be massively faster in nearly every case!

// Base classes.
#include <alpaka/threads/AccThreadsFwd.hpp>
#include <alpaka/threads/WorkExtent.hpp>            // TInterfacedWorkExtent
#include <alpaka/threads/Index.hpp>                 // TInterfacedIndex
#include <alpaka/threads/Atomic.hpp>                // TInterfacedAtomic
#include <alpaka/threads/Barrier.hpp>               // BarrierThreads

// User functionality.
#include <alpaka/host/Memory.hpp>                   // MemCopy
#include <alpaka/threads/Event.hpp>                 // Event
#include <alpaka/threads/Device.hpp>                // Devices

// Specialized templates.
#include <alpaka/interfaces/KernelExecCreator.hpp>  // KernelExecCreator

// Implementation details.
#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>
#ifndef ALPAKA_THREADS_NO_POOL
    #include <alpaka/core/ConcurrentExecutionPool.hpp>  // ConcurrentExecutionPool
#endif

#include <cstddef>                                  // std::size_t
#include <cstdint>                                  // std::uint32_t
#include <vector>                                   // std::vector
#include <thread>                                   // std::thread
#include <map>                                      // std::map
#include <algorithm>                                // std::for_each
#include <array>                                    // std::array
#include <cassert>                                  // assert
#include <stdexcept>                                // std::runtime_error
#include <string>                                   // std::to_string
#ifdef ALPAKA_DEBUG
    #include <iostream>                             // std::cout
#endif

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

// workarounds
#include <boost/predef.h>

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            template<typename TAcceleratedKernel>
            class KernelExecutor;

            //#############################################################################
            //! The base class for all C++11 std::thread accelerated kernels.
            //#############################################################################
            class AccThreads :
                protected TInterfacedWorkExtent,
                protected TInterfacedIndex,
                protected TInterfacedAtomic
            {
            public:
                using MemorySpace = MemorySpaceHost;

                template<typename TAcceleratedKernel>
                friend class alpaka::threads::detail::KernelExecutor;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccThreads() :
                    TInterfacedWorkExtent(),
                    TInterfacedIndex(m_mThreadsToIndices, m_v3uiGridBlockIdx),
                    TInterfacedAtomic()
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                // Has to be explicitly defined because 'std::mutex::mutex(const std::mutex&)' is deleted.
                // Do not copy most members because they are initialized by the executor for each accelerated execution.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccThreads(AccThreads const & ) :
                    TInterfacedWorkExtent(),
                    TInterfacedIndex(m_mThreadsToIndices, m_v3uiGridBlockIdx),
                    TInterfacedAtomic(),
                    m_mThreadsToIndices(),
                    m_v3uiGridBlockIdx(),
                    m_mThreadsToBarrier(),
                    m_mtxBarrier(),
                    m_abarSyncThreads(),
                    m_idMasterThread(),
                    m_vvuiSharedMem(),
                    m_vuiExternalSharedMem()
                {}
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccThreads(AccThreads &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccThreads & operator=(AccThreads const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~AccThreads() noexcept = default;

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
                ALPAKA_FCT_HOST typename alpaka::detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return this->TInterfacedIndex::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<TInterfacedWorkExtent const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST void syncBlockKernels() const
                {
                    auto const idThread(std::this_thread::get_id());
                    auto const itFind(m_mThreadsToBarrier.find(idThread));

                    syncBlockKernels(itFind);
                }
            private:
                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST void syncBlockKernels(std::map<std::thread::id, std::size_t>::iterator const & itFind) const
                {
                    assert(itFind != m_mThreadsToBarrier.end());

                    auto & uiBarIndex(itFind->second);
                    std::size_t const uiBarrierIndex(uiBarIndex % 2);

                    auto & bar(m_abarSyncThreads[uiBarrierIndex]);

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
                    ++uiBarIndex;
                }
            protected:
                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T, std::size_t TuiNumElements>
                ALPAKA_FCT_HOST T * allocBlockSharedMem() const
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    // Assure that all threads have executed the return of the last allocBlockSharedMem function (if there was one before).
                    syncBlockKernels();

                    // Arbitrary decision: The thread that was created first has to allocate the memory.
                    if(m_idMasterThread == std::this_thread::get_id())
                    {
                        // TODO: Optimize: do not initialize the memory on allocation like std::vector does!
                        m_vvuiSharedMem.emplace_back(TuiNumElements);
                    }
                    syncBlockKernels();

                    return reinterpret_cast<T*>(m_vvuiSharedMem.back().data());
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_HOST T * getBlockSharedExternMem() const
                {
                    return reinterpret_cast<T*>(m_vuiExternalSharedMem.data());
                }

#ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
            protected:
#else
            private:
#endif
                // getIdx
                detail::TThreadIdToIndex mutable m_mThreadsToIndices;       //!< The mapping of thread id's to thread indices.
                vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                // syncBlockKernels
                std::size_t mutable m_uiNumKernelsPerBlock;                 //!< The number of kernels per block the barrier has to wait for.
                std::map<
                    std::thread::id,
                    std::size_t> mutable m_mThreadsToBarrier;               //!< The mapping of thread id's to their current barrier.
                std::mutex mutable m_mtxBarrier;
                detail::ThreadBarrier mutable m_abarSyncThreads[2];         //!< The barriers for the synchronization of threads. 
                //!< We have to keep the current and the last barrier because one of the threads can reach the next barrier before a other thread was wakeup from the last one and has checked if it can run.

                // allocBlockSharedMem
                std::thread::id mutable m_idMasterThread;                   //!< The id of the master thread.
                std::vector<std::vector<uint8_t>> mutable m_vvuiSharedMem;  //!< Block shared memory.

                // getBlockSharedExternMem
                std::vector<uint8_t> mutable m_vuiExternalSharedMem;        //!< External block shared memory.
            };

            //#############################################################################
            //! The type given to the ConcurrentExecutionPool for yielding the current thread.
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
            //! The type given to the ConcurrentExecutionPool for returning the current exception.
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
            //! The executor for an accelerated serial kernel.
            //#############################################################################
            template<typename TAcceleratedKernel>
            class KernelExecutor :
                private TAcceleratedKernel
            {
                static_assert(std::is_base_of<IAcc<AccThreads>, TAcceleratedKernel>::value, "The TAcceleratedKernel for the threads::detail::KernelExecutor has to inherit from IAcc<AccThreads>!");

            public:
                using TAcc = AccThreads;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<typename TWorkExtent, typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutor(IWorkExtent<TWorkExtent> const & workExtent, TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...),
#ifdef ALPAKA_THREADS_NO_POOL
                    m_vThreadsInBlock(),
#else
                    m_vFuturesInBlock(),
#endif
                    m_mtxMapInsert()
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccThreads::KernelExecutor()" << std::endl;
#endif
                    (*const_cast<TInterfacedWorkExtent*>(static_cast<TInterfacedWorkExtent const *>(this))) = workExtent;

                    auto const uiNumKernelsPerBlock(workExtent.template getExtent<Block, Kernels, Linear>());
                    /*auto const uiMaxKernelsPerBlock(AccThreads::getExtentBlockKernelsLinearMax());
                    if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                    {
                        throw std::runtime_error(("The given block kernels count '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the threads accelerator!").c_str());
                    }*/

                    this->AccThreads::m_uiNumKernelsPerBlock = uiNumKernelsPerBlock;

                    m_v3uiGridBlocksExtent = workExtent.template getExtent<Grid, Blocks, D3>();
                    m_v3uiBlockKernelsExtent = workExtent.template getExtent<Block, Kernels, D3>();

                    //m_vThreadsInBlock.reserve(uiNumKernelsPerBlock);    // Minimal speedup?
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccThreads::KernelExecutor()" << std::endl;
#endif
                }
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor(KernelExecutor const & other) :
                    TAcceleratedKernel(other),
#ifdef ALPAKA_THREADS_NO_POOL
                    m_vThreadsInBlock(),
#else
                    m_vFuturesInBlock(),
#endif
                    m_mtxMapInsert()
                {}
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor(KernelExecutor && other) :
                    TAcceleratedKernel(std::move(other)),
#ifdef ALPAKA_THREADS_NO_POOL
                    m_vThreadsInBlock(),
#else
                    m_vFuturesInBlock(),
#endif
                    m_mtxMapInsert()
                {}
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor & operator=(KernelExecutor const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~KernelExecutor() noexcept = default;

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<typename... TArgs>
                ALPAKA_FCT_HOST void operator()(TArgs && ... args) const
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccThreads::KernelExecutor::operator()" << std::endl;
#endif
                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(m_v3uiBlockKernelsExtent, std::forward<TArgs>(args)...));
                    this->AccThreads::m_vuiExternalSharedMem.resize(uiBlockSharedExternMemSizeBytes);
#ifdef ALPAKA_DEBUG
                    //std::cout << "GridBlocks: " << v3uiGridBlocksExtent << " BlockKernels: " << v3uiBlockKernelsExtent << std::endl;
#endif

#ifndef ALPAKA_THREADS_NO_POOL
                    auto const uiNumKernelsInBlock(this->AccThreads::getExtent<Block, Kernels, Linear>());
                    // When using the thread pool the threads are yielding because this is faster. 
                    // Using condition variables and going to sleep is very costly for real threads. 
                    // Especially when the time to wait is really short (syncBlockKernels) yielding is much faster.
                    using TPool = alpaka::detail::ConcurrentExecutionPool<
                        std::thread,                // The concurrent execution type.
                        std::promise,               // The promise type.
                        ThreadPoolCurrentException, // The type returning the current exception.
                        ThreadPoolYield             // The type yielding the current concurrent execution.
                    >;
                    TPool pool(uiNumKernelsInBlock, uiNumKernelsInBlock);
#endif
                    // Execute the blocks serially.
                    for(std::uint32_t bz(0); bz<m_v3uiGridBlocksExtent[2]; ++bz)
                    {
                        this->AccThreads::m_v3uiGridBlockIdx[2] = bz;
                        for(std::uint32_t by(0); by<m_v3uiGridBlocksExtent[1]; ++by)
                        {
                            this->AccThreads::m_v3uiGridBlockIdx[1] = by;
                            for(std::uint32_t bx(0); bx<m_v3uiGridBlocksExtent[0]; ++bx)
                            {
                                this->AccThreads::m_v3uiGridBlockIdx[0] = bx;

                                // Execute the kernels in parallel threads.
                                vec<3u> v3uiBlockKernelIdx;
                                for(std::uint32_t tz(0); tz<m_v3uiBlockKernelsExtent[2]; ++tz)
                                {
                                    v3uiBlockKernelIdx[2] = tz;
                                    for(std::uint32_t ty(0); ty<m_v3uiBlockKernelsExtent[1]; ++ty)
                                    {
                                        v3uiBlockKernelIdx[1] = ty;
                                        for(std::uint32_t tx(0); tx<m_v3uiBlockKernelsExtent[0]; ++tx)
                                        {
                                            v3uiBlockKernelIdx[0] = tx;

                                            // Create a thread.
                                            // The v3uiBlockKernelIdx is required to be copied in from the environment because if the thread is immediately suspended the variable is already changed for the next iteration/thread.
//#if BOOST_COMP_MSVC //<= BOOST_VERSION_NUMBER(14, 0, 22310)    MSVC does not compile the std::thread constructor because the type of the member function template is missing the this pointer as first argument.
                                            auto threadKernelFct([this](vec<3u> const v3uiBlockKernelIdx, TArgs ... args) {threadKernel<TArgs...>(v3uiBlockKernelIdx, std::forward<TArgs>(args)...); });
    #ifdef ALPAKA_THREADS_NO_POOL
                                            m_vThreadsInBlock.push_back(std::thread(threadKernelFct, v3uiBlockKernelIdx, args...));
    #else
                                            m_vFuturesInBlock.emplace_back(pool.enqueueTask(threadKernelFct, v3uiBlockKernelIdx, args...));
    #endif
/*#else
    #ifdef ALPAKA_THREADS_NO_POOL
                                            m_vThreadsInBlock.push_back(std::thread(&KernelExecutor::threadKernel<TArgs...>, this, v3uiBlockKernelIdx, args...));
    #else
                                            // FIXME: Currently this does not work!
                                            m_vFuturesInBlock.emplace_back(pool.enqueueTask(&KernelExecutor::threadKernel<TArgs...>, this, v3uiBlockKernelIdx, args...));
    #endif
#endif*/
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
                                this->AccThreads::m_vuiExternalSharedMem.clear();
                            }
                        }
                    }
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccThreads::KernelExecutor::operator()" << std::endl;
#endif
                }
            private:
                //-----------------------------------------------------------------------------
                //! The thread entry point.
                //-----------------------------------------------------------------------------
                template<typename... TArgs>
                ALPAKA_FCT_HOST void threadKernel(vec<3u> const v3uiBlockKernelIdx, TArgs && ... args) const
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
                    std::map<std::thread::id, std::size_t>::iterator itThreadToBarrier;

                    {
                        // The insertion of elements has to be done one thread at a time.
                        std::lock_guard<std::mutex> lock(m_mtxMapInsert);

                        // Save the thread id, and index.
#if BOOST_COMP_GNUC <= BOOST_VERSION_NUMBER(4, 7, 2) // GCC <= 4.7.2 is not standard conformant and has no member emplace.
                        this->AccThreads::m_mThreadsToIndices.emplace(idThread, v3uiBlockKernelIdx);
                        itThreadToBarrier = this->AccThreads::m_mThreadsToBarrier.emplace(idThread, 0).first;
#else
                        this->AccThreads::m_mThreadsToIndices.insert(std::make_pair(idThread, v3uiBlockKernelIdx));
                        itThreadToBarrier = this->AccThreads::m_mThreadsToBarrier.insert(std::make_pair(idThread, 0)).first;
#endif
                    }

                    // Sync all fibers so that the maps with fiber id's are complete and not changed after here.
                    this->AccThreads::syncBlockKernels(itThreadToBarrier);

                    // Execute the kernel itself.
                    this->TAcceleratedKernel::operator()(std::forward<TArgs>(args)...);

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

                vec<3u> m_v3uiGridBlocksExtent;
                vec<3u> m_v3uiBlockKernelsExtent;
            };
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The threads kernel executor builder.
        //#############################################################################
        template<typename TKernel, typename... TKernelConstrArgs>
        class KernelExecCreator<AccThreads, TKernel, TKernelConstrArgs...>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccThreads>::type;
            using KernelExecutorExtent = KernelExecutorExtent<threads::detail::KernelExecutor<TAcceleratedKernel>, TKernelConstrArgs...>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent operator()(TKernelConstrArgs && ... args) const
            {
                return KernelExecutorExtent(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
