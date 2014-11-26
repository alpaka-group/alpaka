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

#include <alpaka/fibers/WorkSize.hpp>               // TInterfacedWorkSize
#include <alpaka/fibers/Index.hpp>                  // TInterfacedIndex
#include <alpaka/fibers/Atomic.hpp>                 // TInterfacedAtomic
#include <alpaka/fibers/Barrier.hpp>                // BarrierFibers

#include <alpaka/fibers/Common.hpp>

#include <alpaka/host/MemorySpace.hpp>              // MemorySpaceHost
#include <alpaka/host/Memory.hpp>                   // MemCopy

#include <alpaka/interfaces/KernelExecCreator.hpp>  // KernelExecCreator

#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>

#include <cstddef>                                  // std::size_t
#include <cstdint>                                  // unit8_t
#include <cassert>                                  // assert
#include <stdexcept>                                // std::except

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

namespace alpaka
{
    namespace fibers
    {
        namespace detail
        {
            template<typename TAcceleratedKernel>
            class KernelExecutor;

            //#############################################################################
            //! The base class for all fibers accelerated kernels.
            //#############################################################################
            class AccFibers :
                protected TInterfacedWorkSize,
                protected TInterfacedIndex,
                protected TInterfacedAtomic
            {
            public:
                using MemorySpace = MemorySpaceHost;

                template<typename TAcceleratedKernel>
                friend class alpaka::fibers::detail::KernelExecutor;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccFibers() :
                    TInterfacedWorkSize(),
                    TInterfacedIndex(m_mFibersToIndices, m_v3uiGridBlockIdx),
                    TInterfacedAtomic()
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                // Has to be explicitly defined because 'std::mutex::mutex(const std::mutex&)' is deleted.
                // Do not copy most members because they are initialized by the executor for each accelerated execution.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccFibers(AccFibers const & ) :
                    TInterfacedWorkSize(),
                    TInterfacedIndex(m_mFibersToIndices, m_v3uiGridBlockIdx),
                    TInterfacedAtomic(),
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
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccFibers(AccFibers &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccFibers & operator=(AccFibers const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~AccFibers() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in each dimension of a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static vec<3u> getSizeBlockKernelsMax()
                {
                    auto const uiSizeBlockKernelsLinearMax(getSizeBlockKernelsLinearMax());
                    return{uiSizeBlockKernelsLinearMax, uiSizeBlockKernelsLinearMax, uiSizeBlockKernelsLinearMax};
                }
                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in a block allowed by the underlying accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::uint32_t getSizeBlockKernelsLinearMax()
                {
                    // FIXME: What is the maximum? Just set a reasonable value?
                    return 1024;    // Magic number.
                }

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
                ALPAKA_FCT_HOST typename alpaka::detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return this->TInterfacedIndex::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<TInterfacedWorkSize const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST void syncBlockKernels() const
                {
                    auto const idFiber(boost::this_fiber::get_id());
                    auto const itFind(m_mFibersToBarrier.find(idFiber));

                    syncBlockKernels(itFind);
                }

            private:
                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST void syncBlockKernels(std::map<boost::fibers::fiber::id, std::size_t>::iterator const & itFind) const
                {
                    assert(itFind != m_mFibersToBarrier.end());

                    auto & uiBarIndex(itFind->second);
                    std::size_t const uiBarrierIndex(uiBarIndex % 2);

                    auto & bar(m_abarSyncFibers[uiBarrierIndex]);

                    // (Re)initialize a barrier if this is the first fiber to reach it.
                    if(bar.getNumFibersToWaitFor() == 0)
                    {
                        // No DCLP required because there can not be an interruption in between the check and the reset.
                        bar.reset(m_uiNumKernelsPerBlock);
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

                    // Assure that all fibers have executed the return of the last allocBlockSharedMem function (if there was one before).
                    syncBlockKernels();

                    // Arbitrary decision: The fiber that was created first has to allocate the memory.
                    if(m_idMasterFiber == boost::this_fiber::get_id())
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

            private:
                // getXxxIdx
                TFiberIdToIndex mutable m_mFibersToIndices;                 //!< The mapping of fibers id's to fibers indices.
                vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                // syncBlockKernels
                std::size_t mutable m_uiNumKernelsPerBlock;                 //!< The number of kernels per block the barrier has to wait for.
                std::map<
                    boost::fibers::fiber::id,
                    std::size_t> mutable m_mFibersToBarrier;                //!< The mapping of fibers id's to their current barrier.
                FiberBarrier mutable m_abarSyncFibers[2];                   //!< The barriers for the synchronization of fibers. 
                //!< We have the keep to current and the last barrier because one of the fibers can reach the next barrier before another fiber was wakeup from the last one and has checked if it can run.

                // allocBlockSharedMem
                boost::fibers::fiber::id mutable m_idMasterFiber;           //!< The id of the master fiber.
                std::vector<std::vector<uint8_t>> mutable m_vvuiSharedMem;  //!< Block shared memory.

                // getBlockSharedExternMem
                std::vector<uint8_t> mutable m_vuiExternalSharedMem;        //!< External block shared memory.
            };

            //#############################################################################
            //! The executor for an accelerated serial kernel.
            //#############################################################################
            template<typename TAcceleratedKernel>
            class KernelExecutor :
                private TAcceleratedKernel
            {
                static_assert(std::is_base_of<IAcc<AccFibers>, TAcceleratedKernel>::value, "The TAcceleratedKernel for the serial::detail::KernelExecutor has to inherit from IAcc<AccFibers>!");
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutor(TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...),
                    m_vFibersInBlock()
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccFibers::KernelExecutor()" << std::endl;
#endif
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccFibers::KernelExecutor()" << std::endl;
#endif
                }
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor(KernelExecutor const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor(KernelExecutor &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor & operator=(KernelExecutor const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~KernelExecutor() noexcept = default;

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<typename TWorkSize, typename... TArgs>
                ALPAKA_FCT_HOST void operator()(IWorkSize<TWorkSize> const & workSize, TArgs && ... args) const
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccFibers::KernelExecutor::operator()" << std::endl;
#endif
                    (*const_cast<TInterfacedWorkSize*>(static_cast<TInterfacedWorkSize const *>(this))) = workSize;

                    auto const uiNumKernelsPerBlock(workSize.template getSize<Block, Kernels, Linear>());
                    auto const uiMaxKernelsPerBlock(AccFibers::getSizeBlockKernelsLinearMax());
                    if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                    {
                        throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the fibers accelerator!").c_str());
                    }

                    this->AccFibers::m_uiNumKernelsPerBlock = uiNumKernelsPerBlock;

                    //m_vFibersInBlock.reserve(uiNumKernelsPerBlock);    // Minimal speedup?

                    auto const v3uiSizeBlockKernels(workSize.template getSize<Block, Kernels, D3>());
                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(v3uiSizeBlockKernels, std::forward<TArgs>(args)...));
                    this->AccFibers::m_vuiExternalSharedMem.resize(uiBlockSharedExternMemSizeBytes);

                    auto const v3uiSizeGridBlocks(workSize.template getSize<Grid, Blocks, D3>());
#ifdef ALPAKA_DEBUG
                    //std::cout << "GridBlocks: " << v3uiSizeGridBlocks << " BlockKernels: " << v3uiSizeBlockKernels << std::endl;
#endif
                    // CUDA programming guide: "Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
                    // This independence requirement allows thread blocks to be scheduled in any order across any number of cores"
                    // TODO: Execute blocks in parallel because kernel executions in a block are cooperatively multi threaded within one real thread.
                    for(std::uint32_t bz(0); bz<v3uiSizeGridBlocks[2]; ++bz)
                    {
                        this->AccFibers::m_v3uiGridBlockIdx[2] = bz;
                        for(std::uint32_t by(0); by<v3uiSizeGridBlocks[1]; ++by)
                        {
                            this->AccFibers::m_v3uiGridBlockIdx[1] = by;
                            for(std::uint32_t bx(0); bx<v3uiSizeGridBlocks[0]; ++bx)
                            {
                                this->AccFibers::m_v3uiGridBlockIdx[0] = bx;

                                vec<3u> v3uiBlockKernelIdx;
                                for(std::uint32_t tz(0); tz<v3uiSizeBlockKernels[2]; ++tz)
                                {
                                    v3uiBlockKernelIdx[2] = tz;
                                    for(std::uint32_t ty(0); ty<v3uiSizeBlockKernels[1]; ++ty)
                                    {
                                        v3uiBlockKernelIdx[1] = ty;
                                        for(std::uint32_t tx(0); tx<v3uiSizeBlockKernels[0]; ++tx)
                                        {
                                            v3uiBlockKernelIdx[0] = tx;

                                            // Create a fiber.
                                            // The v3uiBlockKernelIdx is required to be copied in from the environment because if the fiber is immediately suspended the variable is already changed for the next iteration/thread.
#ifdef _MSC_VER    // MSVC <= 14 do not compile the boost::fibers::fiber constructor because the type of the member function template is missing the this pointer as first argument.
                                            auto fiberKernelFct([this](vec<3u> const v3uiBlockKernelIdx, TArgs ... args) {fiberKernel<TArgs...>(v3uiBlockKernelIdx, std::forward<TArgs>(args)...); });
                                            m_vFibersInBlock.push_back(boost::fibers::fiber(fiberKernelFct, v3uiBlockKernelIdx, args...));
#else
                                            m_vFibersInBlock.push_back(boost::fibers::fiber(&KernelExecutor::fiberKernel<TArgs...>, this, v3uiBlockKernelIdx, args...));
#endif
                                        }
                                    }
                                }
                                // Join all the fibers.
                                std::for_each(m_vFibersInBlock.begin(), m_vFibersInBlock.end(),
                                    [](boost::fibers::fiber & f)
                                {
                                    f.join();
                                }
                                );
                                // Clean up.
                                m_vFibersInBlock.clear();
                                this->AccFibers::m_mFibersToIndices.clear();
                                this->AccFibers::m_mFibersToBarrier.clear();

                                // After a block has been processed, the shared memory can be deleted.
                                this->AccFibers::m_vvuiSharedMem.clear();
                                this->AccFibers::m_vuiExternalSharedMem.clear();
                            }
                        }
                    }
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccFibers::KernelExecutor::operator()" << std::endl;
#endif
                }
            private:
                //-----------------------------------------------------------------------------
                //! The fiber entry point.
                //-----------------------------------------------------------------------------
                template<typename... TArgs>
                ALPAKA_FCT_HOST void fiberKernel(vec<3u> const v3uiBlockKernelIdx, TArgs ... args) const
                {
                    // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                    auto const idFiber(boost::this_fiber::get_id());

                    // Set the master thread id.
                    if(v3uiBlockKernelIdx[0] == 0 && v3uiBlockKernelIdx[1] == 0 && v3uiBlockKernelIdx[2] == 0)
                    {
                        m_idMasterFiber = idFiber;
                    }

                    // We can not use the default syncBlockKernels here because it searches inside m_mFibersToBarrier for the thread id. 
                    // Concurrently searching while others use emplace is unsafe!
                    std::map<boost::fibers::fiber::id, std::size_t>::iterator itFiberToBarrier;

                    // Save the fiber id, and index.
#if ((!defined __GNUC__) || ((__GNUC__ > 4) || (__GNUC__ == 4 && ((__GNUC_MINOR__ > 7) || ((__GNUC_MINOR__ == 7) && (__GNUC_PATCHLEVEL__ == 3)))))) // GCC <= 4.7.2 is not standard conformant and has no member emplace. This works with 4.7.3+.
                    this->AccFibers::m_mFibersToIndices.emplace(idFiber, v3uiBlockKernelIdx);
                    itFiberToBarrier = this->AccFibers::m_mFibersToBarrier.emplace(idFiber, 0).first;
#else
                    this->AccFibers::m_mFibersToIndices.insert(std::pair<boost::fibers::fiber::id, vec<3u>>(idFiber, v3uiBlockKernelIdx));
                    itFiberToBarrier = this->AccFibers::m_mFibersToBarrier.insert(std::pair<boost::fibers::fiber::id, vec<3u>>(idFiber, 0)).first;
#endif
                    // Sync all threads so that the maps with thread id's are complete and not changed after here.
                    this->AccFibers::syncBlockKernels(itFiberToBarrier);

                    // Execute the kernel itself.
                    this->TAcceleratedKernel::operator()(args ...);

                    // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recycled (then duplicate) fiber id!
                    this->AccFibers::syncBlockKernels(itFiberToBarrier);
                }

            private:
                std::vector<boost::fibers::fiber> mutable m_vFibersInBlock; //!< The fibers executing the current block.
            };
        }
    }

    using AccFibers = fibers::detail::AccFibers;

    namespace detail
    {
        //#############################################################################
        //! The fibers kernel executor builder.
        //#############################################################################
        template<typename TKernel, typename... TKernelConstrArgs>
        class KernelExecCreator<AccFibers, TKernel, TKernelConstrArgs...>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccFibers>::type;
            using TKernelExecutor = fibers::detail::KernelExecutor<TAcceleratedKernel>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST TKernelExecutor operator()(TKernelConstrArgs && ... args) const
            {
                return TKernelExecutor(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
