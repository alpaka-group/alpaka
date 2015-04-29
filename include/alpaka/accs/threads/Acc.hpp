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
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu
#include <alpaka/devs/cpu/Event.hpp>        // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>       // StreamCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused
#include <boost/predef.h>                   // workarounds

#include <cassert>                          // assert
#include <memory>                           // std::unique_ptr
#include <thread>                           // std::thread
#include <vector>                           // std::vector

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The threads accelerator.
        //-----------------------------------------------------------------------------
        namespace threads
        {
            //-----------------------------------------------------------------------------
            //! The threads accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                class ExecThreads;

                //#############################################################################
                //! The threads accelerator.
                //!
                //! This accelerator allows parallel kernel execution on a cpu device.
                //! It uses C++11 std::threads to implement the parallelism.
                //#############################################################################
                class AccThreads :
                    protected workdiv::BasicWorkDiv,
                    protected IdxThreads,
                    protected AtomicThreads
                {
                public:
                    friend class ::alpaka::accs::threads::detail::ExecThreads;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccThreads(
                        TWorkDiv const & workDiv) :
                            workdiv::BasicWorkDiv(workDiv),
                            IdxThreads(m_mThreadsToIndices, m_v3uiGridBlockIdx),
                            AtomicThreads(),
                            m_v3uiGridBlockIdx(Vec3<>::zeros()),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads, dim::Dim1>(workDiv)[0u])
                    {}

                public:
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
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccThreads const &) -> AccThreads & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL     // threads/AccThreads.hpp(134): error : the declared exception specification is incompatible with the generated one
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccThreads() = default;
    #else
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccThreads() noexcept = default;
    #endif

                    //-----------------------------------------------------------------------------
                    //! \return The requested indices.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                    -> Vec<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit, TDim>(
                            *static_cast<IdxThreads const *>(this),
                            *static_cast<workdiv::BasicWorkDiv const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                    -> Vec<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                            *static_cast<workdiv::BasicWorkDiv const *>(this));
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
                            addr,
                            value,
                            *static_cast<AtomicThreads const *>(this));
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

    #ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
                protected:
    #else
                private:
    #endif
                    // getIdx
                    detail::ThreadIdToIdxMap mutable m_mThreadsToIndices;       //!< The mapping of thread id's to thread indices.
                    Vec3<> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                    // syncBlockThreads
                    UInt const m_uiNumThreadsPerBlock;                          //!< The number of threads per block the barrier has to wait for.
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
            }
        }
    }

    using AccThreads = accs::threads::detail::AccThreads;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The threads accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::AccThreads;
            };
            //#############################################################################
            //! The threads accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetAccDevProps<
                accs::threads::detail::AccThreads>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::detail::DevCpu const & dev)
                -> alpaka::acc::AccDevProps
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    UInt const uiBlockThreadsCountMax(8u);
#else
                    // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
                    // std::thread::hardware_concurrency  can return 0, so a default for this case?
                    UInt const uiBlockThreadsCountMax(std::thread::hardware_concurrency() * 8u);
#endif
                    return alpaka::acc::AccDevProps(
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_v3uiBlockThreadExtentsMax
                        Vec3<>(
                            uiBlockThreadsCountMax,
                            uiBlockThreadsCountMax,
                            uiBlockThreadsCountMax),
                        // m_v3uiGridBlockExtentsMax
                        Vec3<>::all(std::numeric_limits<Vec3<>::Val>::max()));
                }
            };
            //#############################################################################
            //! The threads accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::threads::detail::AccThreads>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccThreads";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The threads accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::threads::detail::AccThreads>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The threads accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::threads::detail::AccThreads>
            {
                using type = accs::threads::detail::ExecThreads;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The threads accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::threads::detail::AccThreads>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The threads accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::threads::detail::AccThreads>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The threads accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::threads::detail::AccThreads>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
        }
    }
}
