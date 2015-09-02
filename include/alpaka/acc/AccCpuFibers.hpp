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
#include <alpaka/math/MathStl.hpp>              // MathStl
#include <alpaka/block/shared/BlockSharedAllocMasterSync.hpp>   // BlockSharedAllocMasterSync
#include <alpaka/block/sync/BlockSyncFiberIdMapBarrier.hpp>     // BlockSyncFiberIdMapBarrier
#include <alpaka/rand/RandStl.hpp>              // RandStl

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                // dev::traits::DevType
#include <alpaka/exec/Traits.hpp>               // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>               // size::traits::SizeType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu

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
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuFibers;
    }
    namespace acc
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
            public block::shared::BlockSharedAllocMasterSync,
            public block::sync::BlockSyncFiberIdMapBarrier<TSize>,
            public rand::RandStl
        {
        public:
            // Partial specialization with the correct TDim and TSize is not allowed.
            template<
                typename TDim2,
                typename TSize2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::exec::ExecCpuFibers;

        private:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuFibers(
                TWorkDiv const & workDiv) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    idx::gb::IdxGbRef<TDim, TSize>(m_gridBlockIdx),
                    idx::bt::IdxBtRefFiberIdMap<TDim, TSize>(m_fibersToIndices),
                    atomic::AtomicNoOp(),
                    math::MathStl(),
                    block::shared::BlockSharedAllocMasterSync(
                        [this](){block::sync::syncBlockThreads(*this);},
                        [this](){return (m_masterFiberId == boost::this_fiber::get_id());}),
                    block::sync::BlockSyncFiberIdMapBarrier<TSize>(
                        m_threadsPerBlockCount,
                        m_fibersToBarrier),
                    rand::RandStl(),
                    m_gridBlockIdx(Vec<TDim, TSize>::zeros()),
                    m_threadsPerBlockCount(workdiv::getWorkDiv<Block, Threads>(workDiv).prod())
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
            //! \return The pointer to the externally allocated block shared memory.
            //-----------------------------------------------------------------------------
            template<
                typename T>
            ALPAKA_FN_ACC_NO_CUDA auto getBlockSharedExternMem() const
            -> T *
            {
                return reinterpret_cast<T*>(m_externalSharedMem.get());
            }

        private:
            // getIdx
            typename idx::bt::IdxBtRefFiberIdMap<TDim, TSize>::FiberIdToIdxMap mutable m_fibersToIndices;  //!< The mapping of fibers id's to indices.
            alignas(16u) Vec<TDim, TSize> mutable m_gridBlockIdx;    //!< The index of the currently executed block.

            // syncBlockThreads
            TSize const m_threadsPerBlockCount;                         //!< The number of threads per block the barrier has to wait for.
            std::map<
                boost::fibers::fiber::id,
                TSize> mutable m_fibersToBarrier;                      //!< The mapping of fibers id's to their current barrier.
            //!< We have to keep the current and the last barrier because one of the fibers can reach the next barrier before another fiber was wakeup from the last one and has checked if it can run.

            // allocBlockSharedArr
            boost::fibers::fiber::id mutable m_masterFiberId;           //!< The id of the master fiber.

            // getBlockSharedExternMem
            std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_externalSharedMem;  //!< External block shared memory.
        };
    }

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
                acc::AccCpuFibers<TDim, TSize>>
            {
                using type = acc::AccCpuFibers<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU fibers accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccCpuFibers<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    auto const blockThreadsCountMax(static_cast<TSize>(3));
#else
                    auto const blockThreadsCountMax(static_cast<TSize>(4));  // \TODO: What is the maximum? Just set a reasonable value?
#endif
                    return {
                        // m_multiProcessorCount
                        std::max(static_cast<TSize>(1), static_cast<TSize>(std::thread::hardware_concurrency())),   // \TODO: This may be inaccurate.
                        // m_blockThreadsCountMax
                        blockThreadsCountMax,
                        // m_blockThreadExtentsMax
                        Vec<TDim, TSize>::all(blockThreadsCountMax),
                        // m_gridBlockExtentsMax
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
                acc::AccCpuFibers<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
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
                acc::AccCpuFibers<TDim, TSize>>
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
                acc::AccCpuFibers<TDim, TSize>>
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
                acc::AccCpuFibers<TDim, TSize>>
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
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuFibers<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>;
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
                acc::AccCpuFibers<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
