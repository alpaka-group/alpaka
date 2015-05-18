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
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
#include <alpaka/accs/fibers/Idx.hpp>       // IdxFibers
#include <alpaka/accs/fibers/Atomic.hpp>    // AtomicFibers
#include <alpaka/accs/fibers/Barrier.hpp>   // BarrierFibers

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/accs/fibers/Common.hpp>
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu
#include <alpaka/devs/cpu/Event.hpp>        // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>       // StreamCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused
#include <boost/predef.h>                   // workarounds

#include <cassert>                          // assert
#include <memory>                           // std::unique_ptr
#include <vector>                           // std::vector

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The fibers accelerator.
        //-----------------------------------------------------------------------------
        namespace fibers
        {
            //-----------------------------------------------------------------------------
            //! The fibers accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                class ExecFibers;

                //#############################################################################
                //! The fibers accelerator.
                //!
                //! This accelerator allows parallel kernel execution on a cpu device.
                //! It uses boost::fibers to implement the cooperative parallelism.
                //! By using fibers the shared memory can reside in the closest memory/cache available.
                //! Furthermore there is no false sharing between neighboring threads as it is the case in real multi-threading.
                //#############################################################################
                class AccFibers :
                    protected workdiv::BasicWorkDiv,
                    protected IdxFibers,
                    protected AtomicFibers
                {
                public:
                    friend class ::alpaka::accs::fibers::detail::ExecFibers;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccFibers(
                        TWorkDiv const & workDiv) :
                            workdiv::BasicWorkDiv(workDiv),
                            IdxFibers(m_mFibersToIndices, m_v3uiGridBlockIdx),
                            AtomicFibers(),
                            m_v3uiGridBlockIdx(Vec3<>::zeros()),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads, dim::Dim1>(workDiv)[0u])
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccFibers(AccFibers const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccFibers(AccFibers &&) = delete;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccFibers const &) -> AccFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccFibers() noexcept = default;

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
                            *static_cast<IdxFibers const *>(this),
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
                            *static_cast<AtomicFibers const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads() const
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
                    ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads(
                        std::map<boost::fibers::fiber::id, UInt>::iterator const & itFind) const
                    -> void
                    {
                        assert(itFind != m_mFibersToBarrier.end());

                        auto & uiBarrierIdx(itFind->second);
                        std::size_t const uiModBarrierIdx(uiBarrierIdx % 2);

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
                    //! \return Allocates block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T,
                        UInt TuiNumElements>
                    ALPAKA_FCT_ACC_NO_CUDA auto allocBlockSharedMem() const
                    -> T *
                    {
                        static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                        // Assure that all fibers have executed the return of the last allocBlockSharedMem function (if there was one before).
                        syncBlockThreads();

                        // Arbitrary decision: The fiber that was created first has to allocate the memory.
                        if(m_idMasterFiber == boost::this_fiber::get_id())
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
                    FiberIdToIdxMap mutable m_mFibersToIndices;                 //!< The mapping of fibers id's to indices.
                    Vec3<> mutable m_v3uiGridBlockIdx;                          //!< The index of the currently executed block.

                    // syncBlockThreads
                    UInt const m_uiNumThreadsPerBlock;                          //!< The number of threads per block the barrier has to wait for.
                    std::map<
                        boost::fibers::fiber::id,
                        UInt> mutable m_mFibersToBarrier;                       //!< The mapping of fibers id's to their current barrier.
                    FiberBarrier mutable m_abarSyncFibers[2];                   //!< The barriers for the synchronization of fibers.
                    //!< We have to keep the current and the last barrier because one of the fibers can reach the next barrier before another fiber was wakeup from the last one and has checked if it can run.

                    // allocBlockSharedMem
                    boost::fibers::fiber::id mutable m_idMasterFiber;           //!< The id of the master fiber.
                    std::vector<
                        std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };
            }
        }
    }

    using AccFibers = accs::fibers::detail::AccFibers;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The fibers accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::fibers::detail::AccFibers>
            {
                using type = accs::fibers::detail::AccFibers;
            };
            //#############################################################################
            //! The fibers accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetAccDevProps<
                accs::fibers::detail::AccFibers>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::detail::DevCpu const & dev)
                -> alpaka::acc::AccDevProps
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    UInt const uiBlockThreadsCountMax(24u);
#else
                    UInt const uiBlockThreadsCountMax(32u);     // \TODO: What is the maximum? Just set a reasonable value?
#endif
                    return alpaka::acc::AccDevProps(
                        // m_uiMultiProcessorCount
                        std::thread::hardware_concurrency(),    // \TODO: This may be inaccurate.
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_v3uiBlockThreadExtentsMax
                        Vec3<>::all(uiBlockThreadsCountMax),
                        // m_v3uiGridBlockExtentsMax
                        Vec3<>::all(std::numeric_limits<Vec3<>::Val>::max()));
                }
            };
            //#############################################################################
            //! The fibers accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::fibers::detail::AccFibers>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccFibers";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The fibers accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::fibers::detail::AccFibers>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The fibers accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::fibers::detail::AccFibers>
            {
                using type = accs::fibers::detail::ExecFibers;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The fibers accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::fibers::detail::AccFibers>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The fibers accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::fibers::detail::AccFibers>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The fibers accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::fibers::detail::AccFibers>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
        }
    }
}
