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
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers
#include <alpaka/idx/gb/IdxGbRef.hpp>           // IdxGbRef
#include <alpaka/idx/bt/IdxBtRefFiberIdMap.hpp> // IdxBtRefFiberIdMap
#include <alpaka/atomic/AtomicNoOp.hpp>         // AtomicNoOp
#include <alpaka/acc/fibers/Barrier.hpp>        // BarrierFibers
#include <alpaka/math/MathStl.hpp>              // MathStl
#include <alpaka/block/shared/BlockSharedAllocMasterSync.hpp>  // BlockSharedAllocMasterSync

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/dev/Traits.hpp>                // DevType
#include <alpaka/exec/Traits.hpp>               // ExecType
#include <alpaka/size/Traits.hpp>               // size::SizeType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // DevCpu

#include <alpaka/core/Fibers.hpp>

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused
#include <boost/predef.h>                       // workarounds

#include <cassert>                              // assert
#include <memory>                               // std::unique_ptr
#include <typeinfo>                             // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize>
        class ExecCpuFibers;

        namespace fibers
        {
            namespace detail
            {
                template<
                    typename TDim,
                    typename TSize>
                class ExecCpuFibersImpl;
            }
        }
    }
    namespace acc
    {
        //-----------------------------------------------------------------------------
        //! The CPU fibers accelerator.
        //-----------------------------------------------------------------------------
        namespace fibers
        {
            //-----------------------------------------------------------------------------
            //! The CPU fibers accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                //#############################################################################
                //! The CPU fibers accelerator.
                //!
                //! This accelerator allows parallel kernel execution on a CPU device.
                //! It uses boost::fibers to implement the cooperative parallelism.
                //! By using fibers the shared memory can reside in the closest memory/cache available.
                //! Furthermore there is no false sharing between neighboring threads as it is the case in real multi-threading.
                //#############################################################################
                template<
                    typename TDim,
                    typename TSize>
                class AccCpuFibers final :
                    public workdiv::WorkDivMembers<TDim, TSize>,
                    public idx::gb::IdxGbRef<TDim, TSize>,
                    public idx::bt::IdxBtRefFiberIdMap<TDim, TSize>,
                    public atomic::AtomicNoOp,
                    public math::MathStl,
                    public block::shared::BlockSharedAllocMasterSync
                {
                public:
                    friend class ::alpaka::exec::fibers::detail::ExecCpuFibersImpl<TDim, TSize>;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FN_ACC_NO_CUDA AccCpuFibers(
                        TWorkDiv const & workDiv) :
                            workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                            idx::gb::IdxGbRef<TDim, TSize>(m_vuiGridBlockIdx),
                            idx::bt::IdxBtRefFiberIdMap<TDim, TSize>(m_mFibersToIndices),
                            atomic::AtomicNoOp(),
                            math::MathStl(),
                            block::shared::BlockSharedAllocMasterSync(
                                [this](){syncBlockThreads();},
                                [this](){return (m_idMasterFiber == boost::this_fiber::get_id());}),
                            m_vuiGridBlockIdx(Vec<TDim, TSize>::zeros()),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads>(workDiv).prod())
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA AccCpuFibers(AccCpuFibers const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA AccCpuFibers(AccCpuFibers &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuFibers const &) -> AccCpuFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuFibers &&) -> AccCpuFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AccCpuFibers() = default;

                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto syncBlockThreads() const
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
                    ALPAKA_FN_ACC_NO_CUDA auto syncBlockThreads(
                        typename std::map<boost::fibers::fiber::id, TSize>::iterator const & itFind) const
                    -> void
                    {
                        assert(itFind != m_mFibersToBarrier.end());

                        auto & uiBarrierIdx(itFind->second);
                        TSize const uiModBarrierIdx(uiBarrierIdx % 2);

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
                    //! \return The pointer to the externally allocated block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T>
                    ALPAKA_FN_ACC_NO_CUDA auto getBlockSharedExternMem() const
                    -> T *
                    {
                        return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                    }

                private:
                    // getIdx
                    typename idx::bt::IdxBtRefFiberIdMap<TDim, TSize>::FiberIdToIdxMap mutable m_mFibersToIndices;  //!< The mapping of fibers id's to indices.
                    alignas(16u) Vec<TDim, TSize> mutable m_vuiGridBlockIdx;    //!< The index of the currently executed block.

                    // syncBlockThreads
                    TSize const m_uiNumThreadsPerBlock;                         //!< The number of threads per block the barrier has to wait for.
                    std::map<
                        boost::fibers::fiber::id,
                        TSize> mutable m_mFibersToBarrier;                      //!< The mapping of fibers id's to their current barrier.
                    FiberBarrier<TSize> mutable m_abarSyncFibers[2];            //!< The barriers for the synchronization of fibers.
                    //!< We have to keep the current and the last barrier because one of the fibers can reach the next barrier before another fiber was wakeup from the last one and has checked if it can run.

                    // allocBlockSharedArr
                    boost::fibers::fiber::id mutable m_idMasterFiber;           //!< The id of the master fiber.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };
            }
        }
    }

    template<
        typename TDim,
        typename TSize>
    using AccCpuFibers = acc::fibers::detail::AccCpuFibers<TDim, TSize>;

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
            {
                using type = acc::fibers::detail::AccCpuFibers<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU fibers accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    auto const uiBlockThreadsCountMax(static_cast<TSize>(24));
#else
                    auto const uiBlockThreadsCountMax(static_cast<TSize>(32));  // \TODO: What is the maximum? Just set a reasonable value?
#endif
                    return {
                        // m_uiMultiProcessorCount
                        std::max(static_cast<TSize>(1), static_cast<TSize>(std::thread::hardware_concurrency())),   // \TODO: This may be inaccurate.
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim, TSize>::all(uiBlockThreadsCountMax),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max())};
                }
            };
            //#############################################################################
            //! The CPU fibers accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
            {
                ALPAKA_FN_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuFibers<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU fibers accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
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
            //! The CPU fibers accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
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
            //! The CPU fibers accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
            {
                using type = exec::ExecCpuFibers<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::fibers::detail::AccCpuFibers<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
