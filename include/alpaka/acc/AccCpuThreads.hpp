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
#include <alpaka/workdiv/WorkDivMembers.hpp>        // workdiv::WorkDivMembers
#include <alpaka/idx/gb/IdxGbRef.hpp>               // IdxGbRef
#include <alpaka/idx/bt/IdxBtRefThreadIdMap.hpp>    // IdxBtRefThreadIdMap
#include <alpaka/atomic/AtomicStlLock.hpp>          // AtomicStlLock
#include <alpaka/math/MathStl.hpp>                  // MathStl
#include <alpaka/block/shared/BlockSharedAllocMasterSync.hpp>   // BlockSharedAllocMasterSync
#include <alpaka/block/sync/BlockSyncThreadIdMapBarrier.hpp>    // BlockSyncThreadIdMapBarrier

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                    // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                    // dev::traits::DevType
#include <alpaka/exec/Traits.hpp>                   // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>                   // size::traits::SizeType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                    // dev::DevCpu

#include <boost/core/ignore_unused.hpp>             // boost::ignore_unused
#include <boost/predef.h>                           // workarounds

#include <cassert>                                  // assert
#include <memory>                                   // std::unique_ptr
#include <thread>                                   // std::thread
#include <typeinfo>                                 // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuThreads;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU threads accelerator.
        //!
        //! This accelerator allows parallel kernel execution on a CPU device.
        //! It uses C++11 std::thread to implement the parallelism.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class AccCpuThreads final :
            public workdiv::WorkDivMembers<TDim, TSize>,
            public idx::gb::IdxGbRef<TDim, TSize>,
            public idx::bt::IdxBtRefThreadIdMap<TDim, TSize>,
            public atomic::AtomicStlLock,
            public math::MathStl,
            public block::shared::BlockSharedAllocMasterSync,
            public block::sync::BlockSyncThreadIdMapBarrier<TSize>
        {
        public:
            // Partial specialization with the correct TDim and TSize is not allowed.
            template<
                typename TDim2,
                typename TSize2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::exec::ExecCpuThreads;

        private:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuThreads(
                TWorkDiv const & workDiv) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    idx::gb::IdxGbRef<TDim, TSize>(m_gridBlockIdx),
                    idx::bt::IdxBtRefThreadIdMap<TDim, TSize>(m_threadsToIndices),
                    atomic::AtomicStlLock(),
                    math::MathStl(),
                    block::shared::BlockSharedAllocMasterSync(
                        [this](){block::sync::syncBlockThreads(*this);},
                        [this](){return (m_idMasterThread == std::this_thread::get_id());}),
                    block::sync::BlockSyncThreadIdMapBarrier<TSize>(
                        m_threadsPerBlockCount,
                        m_mThreadsToBarrier),
                    m_gridBlockIdx(Vec<TDim, TSize>::zeros()),
                    m_threadsPerBlockCount(workdiv::getWorkDiv<Block, Threads>(workDiv).prod())
            {}

        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuThreads(AccCpuThreads const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuThreads(AccCpuThreads &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuThreads const &) -> AccCpuThreads & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuThreads &&) -> AccCpuThreads & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AccCpuThreads() = default;

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
            std::mutex mutable m_mtxMapInsert;                              //!< The mutex used to secure insertion into the ThreadIdToIdxMap.
            typename idx::bt::IdxBtRefThreadIdMap<TDim, TSize>::ThreadIdToIdxMap mutable m_threadsToIndices;    //!< The mapping of thread id's to indices.
            alignas(16u) Vec<TDim, TSize> mutable m_gridBlockIdx;        //!< The index of the currently executed block.

            // syncBlockThreads
            TSize const m_threadsPerBlockCount;                             //!< The number of threads per block the barrier has to wait for.
            std::map<
                std::thread::id,
                TSize> mutable m_mThreadsToBarrier;                         //!< The mapping of thread id's to their current barrier.

            // allocBlockSharedArr
            std::thread::id mutable m_idMasterThread;                       //!< The id of the master thread.

            // getBlockSharedExternMem
            std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_externalSharedMem;      //!< External block shared memory.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccCpuThreads<TDim, TSize>>
            {
                using type = acc::AccCpuThreads<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU threads accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccCpuThreads<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    auto const blockThreadsCountMax(static_cast<TSize>(8));
#else
                    // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
                    // std::thread::hardware_concurrency can return 0, so 1 is the default case?
                    auto const blockThreadsCountMax(std::max(static_cast<TSize>(1), static_cast<TSize>(std::thread::hardware_concurrency() * 8)));
#endif
                    return {
                        // m_multiProcessorCount
                        static_cast<TSize>(1),
                        // m_blockThreadsCountMax
                        blockThreadsCountMax,
                        // m_blockThreadExtentsMax
                        Vec<TDim, TSize>::all(blockThreadsCountMax),
                        // m_gridBlockExtentsMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max())};
                }
            };
            //#############################################################################
            //! The CPU threads accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccCpuThreads<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuThreads<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccCpuThreads<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU threads accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                acc::AccCpuThreads<TDim, TSize>>
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
            //! The CPU threads accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccCpuThreads<TDim, TSize>>
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
            //! The CPU threads accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuThreads<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccCpuThreads<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
