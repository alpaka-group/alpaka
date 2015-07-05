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
#include <alpaka/workdiv/WorkDivMembers.hpp>        // WorkDivThreads
#include <alpaka/idx/gb/IdxGbRef.hpp>               // IdxGbRef
#include <alpaka/idx/bt/IdxBtRefThreadIdMap.hpp>    // IdxBtRefThreadIdMap
#include <alpaka/atomic/AtomicStlLock.hpp>          // AtomicStlLock
#include <alpaka/acc/threads/Barrier.hpp>           // BarrierThreads
#include <alpaka/math/MathStl.hpp>                  // MathStl
#include <alpaka/block/shared/BlockSharedAllocMasterSync.hpp>  // BlockSharedAllocMasterSync

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                    // AccType
#include <alpaka/dev/Traits.hpp>                    // DevType
#include <alpaka/exec/Traits.hpp>                   // ExecType
#include <alpaka/size/Traits.hpp>                   // size::SizeType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                    // DevCpu

#include <boost/core/ignore_unused.hpp>             // boost::ignore_unused
#include <boost/predef.h>                           // workarounds

#include <cassert>                                  // assert
#include <memory>                                   // std::unique_ptr
#include <thread>                                   // std::thread
#include <mutex>                                    // std::mutex
#include <typeinfo>                                 // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize>
        class ExecCpuThreads;

        namespace threads
        {
            namespace detail
            {
                template<
                    typename TDim,
                    typename TSize>
                class ExecCpuThreadsImpl;
            }
        }
    }
    namespace acc
    {
        //-----------------------------------------------------------------------------
        //! The CPU threads accelerator.
        //-----------------------------------------------------------------------------
        namespace threads
        {
            //-----------------------------------------------------------------------------
            //! The CPU threads accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
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
                    public block::shared::BlockSharedAllocMasterSync
                {
                public:
                    friend class ::alpaka::exec::threads::detail::ExecCpuThreadsImpl<TDim, TSize>;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FN_ACC_NO_CUDA AccCpuThreads(
                        TWorkDiv const & workDiv) :
                            workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                            idx::gb::IdxGbRef<TDim, TSize>(m_vuiGridBlockIdx),
                            idx::bt::IdxBtRefThreadIdMap<TDim, TSize>(m_mThreadsToIndices),
                            atomic::AtomicStlLock(),
                            math::MathStl(),
                            block::shared::BlockSharedAllocMasterSync(
                                [this](){syncBlockThreads();},
                                [this](){return (m_idMasterThread == std::this_thread::get_id());}),
                            m_vuiGridBlockIdx(Vec<TDim, TSize>::zeros()),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads>(workDiv).prod())
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
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto syncBlockThreads() const
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
                    ALPAKA_FN_ACC_NO_CUDA auto syncBlockThreads(
                        typename std::map<std::thread::id, TSize>::iterator const & itFind) const
                    -> void
                    {
                        assert(itFind != m_mThreadsToBarrier.end());

                        auto & uiBarrierIdx(itFind->second);
                        TSize const uiModBarrierIdx(uiBarrierIdx % 2);

                        auto & bar(m_abarSyncThreads[uiModBarrierIdx]);

                        // (Re)initialize a barrier if this is the first thread to reach it.
                        // DCLP: Double checked locking pattern for better performance.
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
                    std::mutex mutable m_mtxMapInsert;                              //!< The mutex used to secure insertion into the ThreadIdToIdxMap.
                    typename idx::bt::IdxBtRefThreadIdMap<TDim, TSize>::ThreadIdToIdxMap mutable m_mThreadsToIndices;    //!< The mapping of thread id's to indices.
                    alignas(16u) Vec<TDim, TSize> mutable m_vuiGridBlockIdx;        //!< The index of the currently executed block.

                    // syncBlockThreads
                    TSize const m_uiNumThreadsPerBlock;                             //!< The number of threads per block the barrier has to wait for.
                    std::map<
                        std::thread::id,
                        TSize> mutable m_mThreadsToBarrier;                         //!< The mapping of thread id's to their current barrier.
                    std::mutex mutable m_mtxBarrier;
                    ThreadBarrier<TSize> mutable m_abarSyncThreads[2];              //!< The barriers for the synchronization of threads.
                    //!< We have to keep the current and the last barrier because one of the threads can reach the next barrier before a other thread was wakeup from the last one and has checked if it can run.

                    // allocBlockSharedArr
                    std::thread::id mutable m_idMasterThread;                       //!< The id of the master thread.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_vuiExternalSharedMem;      //!< External block shared memory.
                };
            }
        }
    }

    template<
        typename TDim,
        typename TSize>
    using AccCpuThreads = acc::threads::detail::AccCpuThreads<TDim, TSize>;

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
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
            {
                using type = acc::threads::detail::AccCpuThreads<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU threads accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    auto const uiBlockThreadsCountMax(static_cast<TSize>(8));
#else
                    // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
                    // std::thread::hardware_concurrency can return 0, so 1 is the default case?
                    auto const uiBlockThreadsCountMax(std::max(static_cast<TSize>(1), static_cast<TSize>(std::thread::hardware_concurrency() * 8)));
#endif
                    return {
                        // m_uiMultiProcessorCount
                        static_cast<TSize>(1),
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim, TSize>::all(uiBlockThreadsCountMax),
                        // m_vuiGridBlockExtentsMax
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
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
            {
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
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
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
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
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
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
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
                typename TSize>
            struct ExecType<
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
            {
                using type = exec::ExecCpuThreads<TDim, TSize>;
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
                acc::threads::detail::AccCpuThreads<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
