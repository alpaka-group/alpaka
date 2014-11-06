/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
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

#include <alpaka/KernelExecutorBuilder.hpp> // KernelExecutorBuilder
#include <alpaka/WorkSize.hpp>              // IWorkSize, WorkSizeDefault

#include <cstddef>                          // std::size_t
#include <cstdint>                          // unit8_t
#include <cassert>                          // assert
#include <stdexcept>                        // std::except

#include <boost/mpl/apply.hpp>              // boost::mpl::apply

// Force the usage of variadics for fibers.
#define BOOST_FIBERS_USE_VARIADIC_FIBER

// Boost fiber: 
// http://olk.github.io/libs/fiber/doc/html/index.html
// https://github.com/olk/boost-fiber
#include <boost/fiber/fiber.hpp>            // boost::fibers::fiber
#include <boost/fiber/operations.hpp>       // boost::this_fiber
#include <boost/fiber/condition.hpp>        // boost::fibers::condition_variable
#include <boost/fiber/mutex.hpp>            // boost::fibers::mutex
//#include <boost/fiber/barrier.hpp>        // boost::fibers::barrier

namespace alpaka
{
    namespace fibers
    {
        namespace detail
		{
			using TFiberIdToIndex = std::map<boost::fibers::fiber::id, vec<3>>;

			//#############################################################################
			//! This class stores the current indices as members.
			//#############################################################################
			class IndexFibers
			{
			public:
				//-----------------------------------------------------------------------------
				//! Constructor.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_CPU_CUDA IndexFibers(
					TFiberIdToIndex & mFibersToIndices,
					vec<3> & v3uiGridBlockIdx) :
					m_mFibersToIndices(mFibersToIndices),
					m_v3uiGridBlockIdx(v3uiGridBlockIdx)
				{
				}

				//-----------------------------------------------------------------------------
				//! Copy-onstructor.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_CPU_CUDA IndexFibers(IndexFibers const & other) = default;

				//-----------------------------------------------------------------------------
				//! \return The index of the currently executed kernel.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_CPU vec<3> getIdxBlockKernel() const
				{
					auto const idFiber(boost::this_fiber::get_id());
					auto const itFind(m_mFibersToIndices.find(idFiber));
					assert(itFind != m_mFibersToIndices.end());

					return itFind->second;
				}
				//-----------------------------------------------------------------------------
				//! \return The index of the block of the currently executed kernel.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_CPU vec<3> getIdxGridBlock() const
				{
					return m_v3uiGridBlockIdx;
				}

			private:
				TFiberIdToIndex const & m_mFibersToIndices;		//!< The mapping of fibers id's to fibers indices.
				vec<3> const & m_v3uiGridBlockIdx;				//!< The index of the currently executed block.
			};

            //#############################################################################
            //! A barrier.
            // NOTE: We do not use the boost::fibers::barrier because it does not support simple resetting.
            //#############################################################################
            class FiberBarrier
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU explicit FiberBarrier(std::size_t uiNumFibersToWaitFor = 0) :
                    m_uiNumFibersToWaitFor{uiNumFibersToWaitFor}
                {
                }

                //-----------------------------------------------------------------------------
                //! Deleted copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU FiberBarrier(FiberBarrier const &) = delete;
                //-----------------------------------------------------------------------------
                //! Deleted assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU FiberBarrier & operator=(FiberBarrier const &) = delete;

                //-----------------------------------------------------------------------------
                //! Waits for all the other fibers to reach the barrier.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU void wait()
                {
                    boost::unique_lock<boost::fibers::mutex> lock(m_mtxBarrier);
					if(--m_uiNumFibersToWaitFor == 0)
                    {
                        m_cvAllFibersReachedBarrier.notify_all();
                    }
                    else
                    {
						m_cvAllFibersReachedBarrier.wait(lock, [this] { return m_uiNumFibersToWaitFor == 0; });
                    }
                }

                //-----------------------------------------------------------------------------
                //! \return The number of fibers to wait for.
                //! NOTE: The value almost always is invalid the time you get it.
                //-----------------------------------------------------------------------------
				ALPAKA_FCT_CPU std::size_t getNumFibersToWaitFor() const
                {
					return m_uiNumFibersToWaitFor;
                }

                //-----------------------------------------------------------------------------
                //! Resets the number of fibers to wait for to the given number.
                //-----------------------------------------------------------------------------
				ALPAKA_FCT_CPU void reset(std::size_t uiNumFibersToWaitFor)
                {
                    //boost::unique_lock<boost::fibers::mutex> lock(m_mtxBarrier);
					m_uiNumFibersToWaitFor = uiNumFibersToWaitFor;
                }

            private:
                boost::fibers::mutex m_mtxBarrier;
				boost::fibers::condition_variable m_cvAllFibersReachedBarrier;
				std::size_t m_uiNumFibersToWaitFor;
            };

			using TPackedIndex = alpaka::detail::IIndex<IndexFibers>;
			using TPackedWorkSize = alpaka::detail::IWorkSize<alpaka::detail::WorkSizeDefault>;

            //#############################################################################
            //! The base class for all fibers accelerated kernels.
            //#############################################################################
            class AccFibers :
                protected TPackedIndex,
                public TPackedWorkSize
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU AccFibers() :
					TPackedIndex(m_mFibersToIndices, m_v3uiGridBlockIdx),
                    TPackedWorkSize()
                {}

				//-----------------------------------------------------------------------------
				//! Copy-constructor.
				// Has to be explicitly defined because 'std::mutex::mutex(const std::mutex&)' is deleted.
				// Do not copy most members because they are initialized by the executor for each accelerated execution.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_CPU AccFibers(AccFibers const & other) :
					TPackedIndex(m_mFibersToIndices, m_v3uiGridBlockIdx),
					TPackedWorkSize(other),
					m_vFibersInBlock(),
					m_mFibersToIndices(),
					m_v3uiGridBlockIdx(),
					m_uiNumKernelsPerBlock(),
					m_mFibersToBarrier(),
					m_abarSyncFibers(),
					m_idMasterFiber(),
					m_vuiSharedMem()
				{}

                //-----------------------------------------------------------------------------
                //! Deleted assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU AccFibers & operator=(AccFibers const &) = delete;

                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in each dimension of a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU static vec<3> getSizeBlockKernelsMax()
                {
                    auto const uiSizeBlockKernelsLinearMax(getSizeBlockKernelsLinearMax());
                    return{uiSizeBlockKernelsLinearMax, uiSizeBlockKernelsLinearMax, uiSizeBlockKernelsLinearMax};
                }
                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in a block allowed by the underlying accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU static std::uint32_t getSizeBlockKernelsLinearMax()
                {
                    // FIXME: What is the maximum? Just set a reasonable value?
                    return 1024;    // Magic number.
                }

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
				ALPAKA_FCT_CPU_CUDA typename alpaka::detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return TPackedIndex::getIdx<TPackedWorkSize, TOrigin, TUnit, TDimensionality>(
                        *static_cast<TPackedWorkSize const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Atomic addition.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_CPU void atomicFetchAdd(T * sum, T summand) const
                {
                    // No mutex required because fibers are thread sequential.
                    auto & rsum = *sum;
                    rsum += summand;
                }
                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU void syncBlockKernels() const
                {
                    auto const idFiber(boost::this_fiber::get_id());
                    auto const itFind(m_mFibersToBarrier.find(idFiber));
                    assert(itFind != m_mFibersToBarrier.end());

                    auto & uiBarIndex(itFind->second);
                    std::size_t const uiBarrierIndex(uiBarIndex % 2);

                    auto & bar(m_abarSyncFibers[uiBarrierIndex]);

                    // (Re)initialize a barrier if this is the first fiber to reach it.
					if(bar.getNumFibersToWaitFor() == 0)
                    {
                        bar.reset(m_uiNumKernelsPerBlock);
                    }

                    // Wait for the barrier.
                    bar.wait();
                    ++uiBarIndex;
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_CPU T * getBlockSharedExternMem() const
                {
                    return reinterpret_cast<T*>(m_vuiSharedMem.data());
                }

            private:
                mutable std::vector<boost::fibers::fiber> m_vFibersInBlock; //!< The fibers executing the current block.

                // getXxxIdx
				mutable TFiberIdToIndex m_mFibersToIndices;                 //!< The mapping of fibers id's to fibers indices.
                mutable vec<3> m_v3uiGridBlockIdx;                          //!< The index of the currently executed block.

                // syncBlockKernels
                mutable std::size_t m_uiNumKernelsPerBlock;                 //!< The number of kernels per block the barrier has to wait for.
                mutable std::map<
                    boost::fibers::fiber::id,
                    std::size_t> m_mFibersToBarrier;                        //!< The mapping of fibers id's to their current barrier.
                mutable FiberBarrier m_abarSyncFibers[2];                   //!< The barriers for the synchronzation of fibers. 
                //!< We have the keep to current and the last barrier because one of the fibers can reach the next barrier before a other fiber was wakeup from the last one and has checked if it can run.
                // getBlockSharedExternMem
                mutable boost::fibers::fiber::id m_idMasterFiber;           //!< The id of the master fiber.
                mutable std::vector<uint8_t> m_vuiSharedMem;                //!< Block shared memory.

            public:
                //#############################################################################
                //! The executor for an accelerated serial kernel.
                // TODO: Check that TAcceleratedKernel inherits from the correct accelerator.
                //#############################################################################
                template<typename TAcceleratedKernel>
                class KernelExecutor :
                    protected TAcceleratedKernel
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
					template<typename TWorkSize2, typename TNonAcceleratedKernel>
					KernelExecutor(TWorkSize2 const & workSize, TNonAcceleratedKernel const & kernel)
					{
						(*static_cast<TPackedWorkSize*>(this)) = workSize;
#ifdef _DEBUG
                        std::cout << "AccFibers::KernelExecutor()" << std::endl;
#endif
                    }

                    //-----------------------------------------------------------------------------
                    //! Executes the accelerated kernel.
                    //-----------------------------------------------------------------------------
                    template<typename... TArgs>
                    void operator()(TArgs && ... args) const
                    {
#ifdef _DEBUG
                        std::cout << "[+] AccFibers::KernelExecutor::operator()" << std::endl;
#endif
                        auto const uiNumKernelsPerBlock(this->AccFibers::template getSize<Block, Kernels, Linear>());
                        auto const uiMaxKernelsPerBlock(AccFibers::getSizeBlockKernelsLinearMax());
                        if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                        {
                            throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the fibers accelerator!").c_str());
                        }

                        auto const v3uiSizeGridBlocks(this->AccFibers::template getSize<Grid, Blocks, D3>());
                        auto const v3uiSizeBlockKernels(this->AccFibers::template getSize<Block, Kernels, D3>());
#ifdef _DEBUG
                        //std::cout << "GridBlocks: " << v3uiSizeGridBlocks << " BlockKernels: " << v3uiSizeBlockKernels << std::endl;
#endif

                        this->AccFibers::m_uiNumKernelsPerBlock = uiNumKernelsPerBlock;

                        this->AccFibers::m_vuiSharedMem.resize(TAcceleratedKernel::getBlockSharedMemSizeBytes(v3uiSizeBlockKernels));

                        // CUDA programming guide: "Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
                        // This independence requirement allows thread blocks to be scheduled in any order across any number of cores"
                        // -> We can execute them serially.
                        for(std::uint32_t bz(0); bz<v3uiSizeGridBlocks[2]; ++bz)
                        {
                            this->AccFibers::m_v3uiGridBlockIdx[2] = bz;
                            for(std::uint32_t by(0); by<v3uiSizeGridBlocks[1]; ++by)
                            {
                                this->AccFibers::m_v3uiGridBlockIdx[1] = by;
                                for(std::uint32_t bx(0); bx<v3uiSizeGridBlocks[0]; ++bx)
                                {
                                    this->AccFibers::m_v3uiGridBlockIdx[0] = bx;

                                    vec<3> v3uiBlockKernelIdx;
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
                                                auto fiberKernelFct([this](vec<3> const v3uiBlockKernelIdx, TArgs ... args){fiberKernel<TArgs...>(v3uiBlockKernelIdx, args...); });
												this->AccFibers::m_vFibersInBlock.push_back(boost::fibers::fiber(fiberKernelFct, v3uiBlockKernelIdx, args...));
#else
                                                this->AccFibers::m_vFibersInBlock.push_back(boost::fibers::fiber(&KernelExecutor::fiberKernel<TArgs...>, this, v3uiBlockKernelIdx, args...));
#endif
                                            }
                                        }
                                    }
                                    // Join all the fibers.
                                    std::for_each(this->AccFibers::m_vFibersInBlock.begin(), this->AccFibers::m_vFibersInBlock.end(),
                                        [](boost::fibers::fiber & f)
                                    {
                                        f.join();
                                    }
                                    );
                                    // Clean up.
                                    this->AccFibers::m_vFibersInBlock.clear();
                                    this->AccFibers::m_mFibersToIndices.clear();
                                    this->AccFibers::m_mFibersToBarrier.clear();

                                    // After a block has been processed, the shared memory can be deleted.
                                    this->AccFibers::m_vuiSharedMem.clear();
                                }
                            }
                        }
#ifdef _DEBUG
                        std::cout << "[-] AccFibers::KernelExecutor::operator()" << std::endl;
#endif
                    }
                private:
                    //-----------------------------------------------------------------------------
                    //! The fiber entry point.
                    //-----------------------------------------------------------------------------
					template<typename... TArgs>
                    void fiberKernel(vec<3> const v3uiBlockKernelIdx, TArgs ... args) const
                    {
                        // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                        auto const idFiber(boost::this_fiber::get_id());

                        {
                            // Save the fiber id, and index.
#ifdef _MSC_VER    // GCC <= 4.7.2 is not standard conformant and has no member emplace. This works with 4.7.3+.
                            this->AccFibers::m_mFibersToIndices.emplace(idFiber, v3uiBlockKernelIdx);
                            this->AccFibers::m_mFibersToBarrier.emplace(idFiber, 0);
#else
                            this->AccFibers::m_mFibersToIndices.insert(std::pair<boost::fibers::fiber::id, vec<3>>(idFiber, v3uiBlockKernelIdx));
                            this->AccFibers::m_mFibersToBarrier.insert(std::pair<boost::fibers::fiber::id, vec<3>>(idFiber, 0));
#endif
                        }

                        // Sync all fibers so that the maps with fiber id's are complete and not changed after here.
                        this->AccFibers::syncBlockKernels();

                        // Execute the kernel itself.
                        this->TAcceleratedKernel::operator()(args ...);

                        // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recyled (then duplicate) fiber id!
                        this->AccFibers::syncBlockKernels();
                    }
                };
            };
        }
    }

    using AccFibers = fibers::detail::AccFibers;

    namespace detail
    {
        //#############################################################################
        //! The fibers kernel executor builder.
        //#############################################################################
        template<typename TKernel, typename TPackedWorkSize>
        class KernelExecutorBuilder<AccFibers, TKernel, TPackedWorkSize>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccFibers>::type;
            using TKernelExecutor = AccFibers::KernelExecutor<TAcceleratedKernel>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            TKernelExecutor operator()(TPackedWorkSize const & workSize, TKernel const & kernel) const
            {
				return TKernelExecutor(workSize, kernel);
            }
        };
    }
}