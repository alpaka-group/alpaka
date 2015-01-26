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

//#define ALPAKA_FIBERS_NO_POOL     // Define this to recreate all of the fibers between executing blocks.
                                    // NOTE: Using the fiber pool should be massively faster in nearly every case!

// Base classes.
#include <alpaka/fibers/AccFibersFwd.hpp>
#include <alpaka/fibers/WorkDiv.hpp>                // WorkDivFibers
#include <alpaka/fibers/Idx.hpp>                    // IdxFibers
#include <alpaka/fibers/Atomic.hpp>                 // AtomicFibers
#include <alpaka/fibers/Barrier.hpp>                // BarrierFibers

// User functionality.
#include <alpaka/host/Mem.hpp>                      // MemCopy
#include <alpaka/fibers/Stream.hpp>                 // StreamFibers
#include <alpaka/fibers/Event.hpp>                  // EventFibers
#include <alpaka/fibers/Device.hpp>                 // Devices

// Specialized traits.
#include <alpaka/core/KernelExecCreator.hpp>        // KernelExecCreator

// Implementation details.
#include <alpaka/fibers/Common.hpp>
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>
#ifndef ALPAKA_FIBERS_NO_POOL
    #include <alpaka/core/ConcurrentExecPool.hpp>  // ConcurrentExecPool
#endif
#include <alpaka/core/WorkDivHelpers.hpp>           // isValidWorkDiv

#include <cstddef>                                  // std::size_t
#include <cstdint>                                  // std::uint32_t
#include <cassert>                                  // assert
#include <stdexcept>                                // std::except

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

// Workarounds.
#include <boost/predef.h>

namespace alpaka
{
    namespace fibers
    {
        namespace detail
        {
            template<
                typename TAcceleratedKernel>
            class KernelExecutorFibers;

            //#############################################################################
            //! The fibers accelerator.
            //!
            //! This accelerator allows parallel kernel execution on the host.
            //! It uses boost::fibers to implement the cooperative parallelism.
            //! By using fibers the shared memory can reside in the closest memory/cache available.
            //! Furthermore there is no false sharing between neighboring kernels as it is the case in real multi-threading. 
            //#############################################################################
            class AccFibers :
                protected WorkDivFibers,
                protected IdxFibers,
                protected AtomicFibers
            {
            public:
                using MemSpace = mem::MemSpaceHost;

                template<
                    typename TAcceleratedKernel>
                friend class KernelExecutorFibers;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccFibers() :
                    WorkDivFibers(),
                    IdxFibers(m_mFibersToIndices, m_v3uiGridBlockIdx),
                    AtomicFibers()
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                // Do not copy most members because they are initialized by the executor for each accelerated execution.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccFibers(AccFibers const & ) :
                    WorkDivFibers(),
                    IdxFibers(m_mFibersToIndices, m_v3uiGridBlockIdx),
                    AtomicFibers(),
                    m_mFibersToIndices(),
                    m_v3uiGridBlockIdx(),
                    m_uiNumKernelsPerBlock(),
                    m_mFibersToBarrier(),
                    m_abarSyncFibers(),
                    m_idMasterFiber(),
                    m_vvuiSharedMem(),
                    m_vuiExternalSharedMem()
                {}
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccFibers(AccFibers &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccFibers & operator=(AccFibers const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AccFibers() noexcept = default;

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested indices.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin, 
                    typename TUnit, 
                    typename TDimensionality = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA typename dim::DimToVecT<TDimensionality> getIdx() const
                {
                    return idx::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<IdxFibers const *>(this),
                        *static_cast<WorkDivFibers const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! \return The requested extents.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin,
                    typename TUnit,
                    typename TDimensionality = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA typename dim::DimToVecT<TDimensionality> getWorkDiv() const
                {
                    return workdiv::getWorkDiv<TOrigin, TUnit, TDimensionality>(
                        *static_cast<WorkDivFibers const *>(this));
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
                        *static_cast<AtomicFibers const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockKernels() const
                {
                    auto const idFiber(boost::this_fiber::get_id());
                    auto const itFind(m_mFibersToBarrier.find(idFiber));

                    syncBlockKernels(itFind);
                }

            private:
                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockKernels(
                    std::map<boost::fibers::fiber::id, std::size_t>::iterator const & itFind) const
                {
                    assert(itFind != m_mFibersToBarrier.end());

                    auto & uiBarIdx(itFind->second);
                    std::size_t const uiBarrierIdx(uiBarIdx % 2);

                    auto & bar(m_abarSyncFibers[uiBarrierIdx]);

                    // (Re)initialize a barrier if this is the first fiber to reach it.
                    if(bar.getNumFibersToWaitFor() == 0)
                    {
                        // No DCLP required because there can not be an interruption in between the check and the reset.
                        bar.reset(m_uiNumKernelsPerBlock);
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
                    std::size_t TuiNumElements>
                ALPAKA_FCT_ACC_NO_CUDA T * allocBlockSharedMem() const
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    // Assure that all fibers have executed the return of the last allocBlockSharedMem function (if there was one before).
                    syncBlockKernels();

                    // Arbitrary decision: The fiber that was created first has to allocate the memory.
                    if(m_idMasterFiber == boost::this_fiber::get_id())
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
                // getXxxIdx
                FiberIdToIdxMap mutable m_mFibersToIndices;                 //!< The mapping of fibers id's to fibers indices.
                Vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                // syncBlockKernels
                std::size_t mutable m_uiNumKernelsPerBlock;                 //!< The number of kernels per block the barrier has to wait for.
                std::map<
                    boost::fibers::fiber::id,
                    std::size_t> mutable m_mFibersToBarrier;                //!< The mapping of fibers id's to their current barrier.
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
                ALPAKA_FCT_ACC_NO_CUDA static void yield()
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
            template<
                typename TAcceleratedKernel>
            class KernelExecutorFibers :
                private TAcceleratedKernel,
                private IAcc<AccFibers>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv, 
                    typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutorFibers(
                    TWorkDiv const & workDiv, 
                    StreamFibers const &,
                    TKernelConstrArgs && ... args):
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...),
#ifdef ALPAKA_FIBERS_NO_POOL
                    m_vFibersInBlock()
#else
                    m_vFuturesInBlock()
#endif
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccFibers::KernelExecutorFibers()" << std::endl;
#endif
                    (*static_cast<WorkDivFibers *>(this)) = workDiv;

                    this->AccFibers::m_uiNumKernelsPerBlock = workdiv::getWorkDiv<Block, Kernels, dim::Dim1>(workDiv)[0];
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccFibers::KernelExecutorFibers()" << std::endl;
#endif
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorFibers(KernelExecutorFibers const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorFibers(KernelExecutorFibers &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorFibers & operator=(KernelExecutorFibers const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                ALPAKA_FCT_HOST virtual ~KernelExecutorFibers() = default;
#else
                ALPAKA_FCT_HOST virtual ~KernelExecutorFibers() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<
                    typename... TArgs>
                ALPAKA_FCT_HOST void operator()(
                    TArgs && ... args) const
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccFibers::KernelExecutorFibers::operator()" << std::endl;
#endif
                    Vec<3u> const v3uiGridBlocksExtents(this->AccFibers::getWorkDiv<Grid, Blocks, dim::Dim3>());
                    Vec<3u> const v3uiBlockKernelsExtents(this->AccFibers::getWorkDiv<Block, Kernels, dim::Dim3>());

                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(v3uiBlockKernelsExtents, std::forward<TArgs>(args)...));
                    this->AccFibers::m_vuiExternalSharedMem.reset(
                        new uint8_t[uiBlockSharedExternMemSizeBytes]);

#ifndef ALPAKA_FIBERS_NO_POOL
                    auto const uiNumKernelsInBlock(this->AccFibers::getWorkDiv<Block, Kernels, dim::Dim1>());
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
                    FiberPool pool(uiNumKernelsInBlock[0], uiNumKernelsInBlock[0]);
#endif

                    // Execute the blocks serially.
                    for(std::uint32_t bz(0); bz<v3uiGridBlocksExtents[2]; ++bz)
                    {
                        this->AccFibers::m_v3uiGridBlockIdx[2] = bz;
                        for(std::uint32_t by(0); by<v3uiGridBlocksExtents[1]; ++by)
                        {
                            this->AccFibers::m_v3uiGridBlockIdx[1] = by;
                            for(std::uint32_t bx(0); bx<v3uiGridBlocksExtents[0]; ++bx)
                            {
                                this->AccFibers::m_v3uiGridBlockIdx[0] = bx;

                                // Execute the kernels in parallel using cooperative multi-threading.
                                Vec<3u> v3uiBlockKernelIdx;
                                for(std::uint32_t tz(0); tz<v3uiBlockKernelsExtents[2]; ++tz)
                                {
                                    v3uiBlockKernelIdx[2] = tz;
                                    for(std::uint32_t ty(0); ty<v3uiBlockKernelsExtents[1]; ++ty)
                                    {
                                        v3uiBlockKernelIdx[1] = ty;
                                        for(std::uint32_t tx(0); tx<v3uiBlockKernelsExtents[0]; ++tx)
                                        {
                                            v3uiBlockKernelIdx[0] = tx;

                                            // Create a fiber.
                                            // The v3uiBlockKernelIdx is required to be copied in from the environment because if the fiber is immediately suspended the variable is already changed for the next iteration/thread.
//#if BOOST_COMP_MSVC //<= BOOST_VERSION_NUMBER(14, 0, 22310)    MSVC does not compile the boost::fibers::fiber constructor because the type of the member function template is missing the this pointer as first argument.
                                            auto fiberKernelFct([this](Vec<3u> const v3uiBlockKernelIdx, TArgs ... args) {fiberKernel<TArgs...>(v3uiBlockKernelIdx, std::forward<TArgs>(args)...); });
    #ifdef ALPAKA_FIBERS_NO_POOL
                                            m_vFibersInBlock.push_back(boost::fibers::fiber(fiberKernelFct, v3uiBlockKernelIdx, args...));
    #else
                                            m_vFuturesInBlock.emplace_back(pool.enqueueTask(fiberKernelFct, v3uiBlockKernelIdx, args...));
    #endif
/*#else
    #ifdef ALPAKA_FIBERS_NO_POOL
                                            m_vFibersInBlock.push_back(boost::fibers::fiber(&KernelExecutorFibers::fiberKernel<TArgs...>, this, v3uiBlockKernelIdx, args...));
    #else
                                            // FIXME: Currently this does not work!
                                            m_vFuturesInBlock.emplace_back(pool.enqueueTask(&KernelExecutorFibers::fiberKernel<TArgs...>, this, v3uiBlockKernelIdx, args...));
    #endif
#endif*/
                                        }
                                    }
                                }
#ifdef ALPAKA_FIBERS_NO_POOL
                                // Join all the fibers.
                                std::for_each(m_vFibersInBlock.begin(), m_vFibersInBlock.end(),
                                    [](boost::fibers::fiber & f)
                                    {
                                        f.join();
                                    }
                                );
                                // Clean up.
                                m_vFibersInBlock.clear();
#else
                                // Join all the threads.
                                std::for_each(m_vFuturesInBlock.begin(), m_vFuturesInBlock.end(),
                                    [](boost::fibers::future<void> & t)
                                    {
                                        t.wait();
                                    }
                                );
                                // Clean up.
                                m_vFuturesInBlock.clear();
#endif
                                this->AccFibers::m_mFibersToIndices.clear();
                                this->AccFibers::m_mFibersToBarrier.clear();

                                // After a block has been processed, the shared memory can be deleted.
                                this->AccFibers::m_vvuiSharedMem.clear();
                            }
                        }
                    }
                    // After all blocks have been processed, the external shared memory can be deleted.
                    this->AccFibers::m_vuiExternalSharedMem.reset();
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccFibers::KernelExecutorFibers::operator()" << std::endl;
#endif
                }
            private:
                //-----------------------------------------------------------------------------
                //! The fiber entry point.
                //-----------------------------------------------------------------------------
                template<
                    typename... TArgs>
                ALPAKA_FCT_HOST void fiberKernel(
                    Vec<3u> const v3uiBlockKernelIdx, 
                    TArgs && ... args) const
                {
                    // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                    auto const idFiber(boost::this_fiber::get_id());

                    // Set the master thread id.
                    if(v3uiBlockKernelIdx[0] == 0 && v3uiBlockKernelIdx[1] == 0 && v3uiBlockKernelIdx[2] == 0)
                    {
                        this->AccFibers::m_idMasterFiber = idFiber;
                    }

                    // We can not use the default syncBlockKernels here because it searches inside m_mFibersToBarrier for the thread id. 
                    // Concurrently searching while others use emplace is unsafe!
                    std::map<boost::fibers::fiber::id, std::size_t>::iterator itFiberToBarrier;

                    // Save the fiber id, and index.
#if BOOST_COMP_GNUC <= BOOST_VERSION_NUMBER(4, 7, 2) // GCC <= 4.7.2 is not standard conformant and has no member emplace.
                    this->AccFibers::m_mFibersToIndices.emplace(idFiber, v3uiBlockKernelIdx);
                    itFiberToBarrier = this->AccFibers::m_mFibersToBarrier.emplace(idFiber, 0).first;
#else
                    this->AccFibers::m_mFibersToIndices.insert(std::pair<boost::fibers::fiber::id, Vec<3u>>(idFiber, v3uiBlockKernelIdx));
                    itFiberToBarrier = this->AccFibers::m_mFibersToBarrier.insert(std::pair<boost::fibers::fiber::id, Vec<3u>>(idFiber, 0)).first;
#endif
                    // Sync all threads so that the maps with thread id's are complete and not changed after here.
                    this->AccFibers::syncBlockKernels(itFiberToBarrier);

                    // Execute the kernel itself.
                    this->TAcceleratedKernel::operator()(
                        (*static_cast<IAcc<AccFibers> const *>(this)),
                        std::forward<TArgs>(args)...);

                    // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recycled (then duplicate) fiber id!
                    this->AccFibers::syncBlockKernels(itFiberToBarrier);
                }

            private:
#ifdef ALPAKA_FIBERS_NO_POOL
                std::vector<boost::fibers::fiber> mutable m_vFibersInBlock;         //!< The fibers executing the current block.
#else
                std::vector<boost::fibers::future<void>> mutable m_vFuturesInBlock; //!< The futures of the fibers in the current block.
#endif
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The fibers accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename AcceleratedKernel>
            struct GetAcc<
                fibers::detail::KernelExecutorFibers<AcceleratedKernel >>
            {
                using type = AccFibers;
            };
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The fibers accelerator kernel executor builder.
        //#############################################################################
        template<
            typename TKernel, 
            typename... TKernelConstrArgs>
        class KernelExecCreator<
            AccFibers, 
            TKernel, 
            TKernelConstrArgs...>
        {
        public:
            using AcceleratedKernel = typename boost::mpl::apply<TKernel, AccFibers>::type;
            using AcceleratedKernelExecutorExtent = KernelExecutorExtent<fibers::detail::KernelExecutorFibers<AcceleratedKernel>, TKernelConstrArgs...>;

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
