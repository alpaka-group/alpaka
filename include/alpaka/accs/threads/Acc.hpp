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
#include <alpaka/core/BasicWorkDiv.hpp>     // WorkDivThreads
#include <alpaka/accs/threads/Idx.hpp>      // IdxThreads
#include <alpaka/accs/threads/Atomic.hpp>   // AtomicThreads
#include <alpaka/accs/threads/Barrier.hpp>  // BarrierThreads

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Dev.hpp>            // DevType

// Implementation details.
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused
#include <boost/predef.h>                   // workarounds

#include <cassert>                          // assert
#include <memory>                           // std::unique_ptr
#include <thread>                           // std::thread
#include <vector>                           // std::vector
#include <mutex>                            // std::mutex

namespace alpaka
{
    namespace accs
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
                template<
                    typename TDim>
                class ExecCpuThreads;
                template<
                    typename TDim>
                class ExecCpuThreadsImpl;

                //#############################################################################
                //! The CPU threads accelerator.
                //!
                //! This accelerator allows parallel kernel execution on a CPU device.
                //! It uses C++11 std::thread to implement the parallelism.
                //#############################################################################
                template<
                    typename TDim>
                class AccCpuThreads final :
                    protected workdiv::BasicWorkDiv<TDim>,
                    protected IdxThreads<TDim>,
                    protected AtomicThreads
                {
                public:
                    friend class ::alpaka::accs::threads::detail::ExecCpuThreadsImpl<TDim>;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuThreads(
                        TWorkDiv const & workDiv) :
                            workdiv::BasicWorkDiv<TDim>(workDiv),
                            IdxThreads<TDim>(m_mThreadsToIndices, m_vuiGridBlockIdx),
                            AtomicThreads(),
                            m_vuiGridBlockIdx(Vec<TDim>::zeros()),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads>(workDiv).prod())
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuThreads(AccCpuThreads const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuThreads(AccCpuThreads &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuThreads const &) -> AccCpuThreads & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuThreads &&) -> AccCpuThreads & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA ~AccCpuThreads() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The requested indices.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit>
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                    -> Vec<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit>(
                            *static_cast<IdxThreads<TDim> const *>(this),
                            *static_cast<workdiv::BasicWorkDiv<TDim> const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit>
                    ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                    -> Vec<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit>(
                            *static_cast<workdiv::BasicWorkDiv<TDim> const *>(this));
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
                            *static_cast<AtomicThreads const *>(this),
                            addr,
                            value);
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

                private:
                    // getIdx
                    std::mutex mutable m_mtxMapInsert;                              //!< The mutex used to secure insertion into the ThreadIdToIdxMap.
                    typename IdxThreads<TDim>::ThreadIdToIdxMap mutable m_mThreadsToIndices;    //!< The mapping of thread id's to indices.
                    Vec<TDim> mutable m_vuiGridBlockIdx;                            //!< The index of the currently executed block.

                    // syncBlockThreads
                    UInt const m_uiNumThreadsPerBlock;                              //!< The number of threads per block the barrier has to wait for.
                    std::map<
                        std::thread::id,
                        UInt> mutable m_mThreadsToBarrier;                          //!< The mapping of thread id's to their current barrier.
                    std::mutex mutable m_mtxBarrier;
                    detail::ThreadBarrier mutable m_abarSyncThreads[2];             //!< The barriers for the synchronization of threads.
                    //!< We have to keep the current and the last barrier because one of the threads can reach the next barrier before a other thread was wakeup from the last one and has checked if it can run.

                    // allocBlockSharedMem
                    std::thread::id mutable m_idMasterThread;                       //!< The id of the master thread.
                    std::vector<
                        std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;        //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;      //!< External block shared memory.
                };
            }
        }
    }

    template<
        typename TDim>
    using AccCpuThreads = accs::threads::detail::AccCpuThreads<TDim>;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU threads accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::threads::detail::AccCpuThreads<TDim>>
            {
                using type = accs::threads::detail::AccCpuThreads<TDim>;
            };
            //#############################################################################
            //! The CPU threads accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                accs::threads::detail::AccCpuThreads<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    UInt const uiBlockThreadsCountMax(8u);
#else
                    // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
                    // std::thread::hardware_concurrency can return 0, so 1 is the default case?
                    UInt const uiBlockThreadsCountMax(std::max(1u, std::thread::hardware_concurrency() * 8u));
#endif
                    return {
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim>::all(uiBlockThreadsCountMax),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim>::all(std::numeric_limits<typename Vec<TDim>::Val>::max())};
                }
            };
            //#############################################################################
            //! The CPU threads accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccName<
                accs::threads::detail::AccCpuThreads<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuThreads<" + std::to_string(TDim::value) + ">";
                }
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU threads accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::threads::detail::AccCpuThreads<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU threads accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::threads::detail::AccCpuThreads<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU threads accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::threads::detail::AccCpuThreads<TDim>>
            {
                using type = TDim;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU threads accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::threads::detail::AccCpuThreads<TDim>>
            {
                using type = accs::threads::detail::ExecCpuThreads<TDim>;
            };
        }
    }
}
