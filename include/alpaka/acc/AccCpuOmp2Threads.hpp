/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#if _OPENMP < 200203
    #error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/bt/IdxBtOmp.hpp>
#include <alpaka/atomic/AtomicStlLock.hpp>
#include <alpaka/atomic/AtomicOmpCritSec.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStl.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynBoostAlignedAlloc.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>
#include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>
#include <alpaka/rand/RandStl.hpp>
#include <alpaka/time/TimeOmp.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>

#include <alpaka/core/OpenMp.hpp>

#include <boost/core/ignore_unused.hpp>

#include <limits>
#include <typeinfo>

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOmp2Threads;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 thread accelerator.
        //!
        //! This accelerator allows parallel kernel execution on a CPU device.
        //! It uses OpenMP 2.0 to implement the block thread parallelism.
        template<
            typename TDim,
            typename TIdx>
        class AccCpuOmp2Threads final :
            public workdiv::WorkDivMembers<TDim, TIdx>,
            public idx::gb::IdxGbRef<TDim, TIdx>,
            public idx::bt::IdxBtOmp<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicStlLock<16>,   // grid atomics
                atomic::AtomicOmpCritSec,    // block atomics
                atomic::AtomicOmpCritSec     // thread atomics
            >,
            public math::MathStl,
            public block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc,
            public block::shared::st::BlockSharedMemStMasterSync,
            public block::sync::BlockSyncBarrierOmp,
            public rand::RandStl,
            public time::TimeOmp
        {
        public:
            // Partial specialization with the correct TDim and TIdx is not allowed.
            template<
                typename TDim2,
                typename TSize2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::exec::ExecCpuOmp2Threads;

        private:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(
                TWorkDiv const & workDiv,
                TIdx const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivMembers<TDim, TIdx>(workDiv),
                    idx::gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx),
                    idx::bt::IdxBtOmp<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicStlLock<16>,// atomics between grids
                        atomic::AtomicOmpCritSec, // atomics between blocks
                        atomic::AtomicOmpCritSec  // atomics between threads
                    >(),
                    math::MathStl(),
                    block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    block::shared::st::BlockSharedMemStMasterSync(
                        [this](){block::sync::syncBlockThreads(*this);},
                        [](){return (::omp_get_thread_num() == 0);}),
                    block::sync::BlockSyncBarrierOmp(),
                    rand::RandStl(),
                    time::TimeOmp(),
                    m_gridBlockIdx(vec::Vec<TDim, TIdx>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(AccCpuOmp2Threads const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(AccCpuOmp2Threads &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Threads const &) -> AccCpuOmp2Threads & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Threads &&) -> AccCpuOmp2Threads & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccCpuOmp2Threads() = default;

        private:
            // getIdx
            vec::Vec<TDim, TIdx> mutable m_gridBlockIdx;  //!< The index of the currently executed block.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccCpuOmp2Threads<TDim, TIdx>>
            {
                using type = acc::AccCpuOmp2Threads<TDim, TIdx>;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccCpuOmp2Threads<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TIdx>
                {
                    boost::ignore_unused(dev);

                    // m_blockThreadCountMax
#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(static_cast<TIdx>(4));
#else
                    auto const blockThreadCountMax(static_cast<TIdx>(omp::getMaxOmpThreads()));
#endif
                    return {
                        // m_multiProcessorCount
                        static_cast<TIdx>(1),
                        // m_gridBlockExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        vec::Vec<TDim, TIdx>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()};
                }
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccCpuOmp2Threads<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp2Threads<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccCpuOmp2Threads<TDim, TIdx>>
            {
                using type = dev::DevCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccCpuOmp2Threads<TDim, TIdx>>
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
            //! The CPU OpenMP 2.0 thread accelerator executor type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuOmp2Threads<TDim, TIdx>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread executor platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccCpuOmp2Threads<TDim, TIdx>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccCpuOmp2Threads<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
