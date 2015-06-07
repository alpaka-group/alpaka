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
#include <alpaka/traits/Dev.hpp>            // DevType

// Implementation details.
#include <alpaka/accs/fibers/Common.hpp>
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused
#include <boost/predef.h>                   // workarounds
#include <boost/align.hpp>                  // boost::aligned_alloc

#include <cassert>                          // assert
#include <memory>                           // std::unique_ptr
#include <vector>                           // std::vector

namespace alpaka
{
    namespace accs
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
                template<
                    typename TDim>
                class ExecCpuFibers;
                template<
                    typename TDim>
                class ExecCpuFibersImpl;

                //#############################################################################
                //! The CPU fibers accelerator.
                //!
                //! This accelerator allows parallel kernel execution on a CPU device.
                //! It uses boost::fibers to implement the cooperative parallelism.
                //! By using fibers the shared memory can reside in the closest memory/cache available.
                //! Furthermore there is no false sharing between neighboring threads as it is the case in real multi-threading.
                //#############################################################################
                template<
                    typename TDim>
                class AccCpuFibers final :
                    protected workdiv::BasicWorkDiv<TDim>,
                    protected IdxFibers<TDim>,
                    protected AtomicFibers
                {
                public:
                    friend class ::alpaka::accs::fibers::detail::ExecCpuFibersImpl<TDim>;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuFibers(
                        TWorkDiv const & workDiv) :
                            workdiv::BasicWorkDiv<TDim>(workDiv),
                            IdxFibers<TDim>(m_mFibersToIndices, m_vuiGridBlockIdx),
                            AtomicFibers(),
                            m_vuiGridBlockIdx(Vec<TDim>::zeros()),
                            m_uiNumThreadsPerBlock(workdiv::getWorkDiv<Block, Threads>(workDiv).prod())
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuFibers(AccCpuFibers const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuFibers(AccCpuFibers &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuFibers const &) -> AccCpuFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuFibers &&) -> AccCpuFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~AccCpuFibers() = default;

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
                            *static_cast<IdxFibers<TDim> const *>(this),
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
                            *static_cast<AtomicFibers const *>(this),
                            addr,
                            value);
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
                            m_vvuiSharedMem.emplace_back(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(16u, sizeof(T) * TuiNumElements)));
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
                    typename IdxFibers<TDim>::FiberIdToIdxMap mutable m_mFibersToIndices;   //!< The mapping of fibers id's to indices.
                    alignas(16u) Vec<TDim> mutable m_vuiGridBlockIdx;                       //!< The index of the currently executed block.

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
                        std::unique_ptr<uint8_t, boost::alignment::aligned_delete>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };
            }
        }
    }

    template<
        typename TDim>
    using AccCpuFibers = accs::fibers::detail::AccCpuFibers<TDim>;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU fibers accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::fibers::detail::AccCpuFibers<TDim>>
            {
                using type = accs::fibers::detail::AccCpuFibers<TDim>;
            };
            //#############################################################################
            //! The CPU fibers accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                accs::fibers::detail::AccCpuFibers<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    UInt const uiBlockThreadsCountMax(24u);
#else
                    UInt const uiBlockThreadsCountMax(32u);     // \TODO: What is the maximum? Just set a reasonable value?
#endif
                    return {
                        // m_uiMultiProcessorCount
                        std::max(1u, std::thread::hardware_concurrency()),    // \TODO: This may be inaccurate.
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim>::all(uiBlockThreadsCountMax),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim>::all(std::numeric_limits<typename Vec<TDim>::Val>::max())};
                }
            };
            //#############################################################################
            //! The CPU fibers accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccName<
                accs::fibers::detail::AccCpuFibers<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuFibers<" + std::to_string(TDim::value) + ">";
                }
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU fibers accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::fibers::detail::AccCpuFibers<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU fibers accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::fibers::detail::AccCpuFibers<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU fibers accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::fibers::detail::AccCpuFibers<TDim>>
            {
                using type = TDim;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU fibers accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::fibers::detail::AccCpuFibers<TDim>>
            {
                using type = accs::fibers::detail::ExecCpuFibers<TDim>;
            };
        }
    }
}
