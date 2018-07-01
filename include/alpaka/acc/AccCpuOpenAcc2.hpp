/**
* \file
* Copyright 2018 Benjamin Worpitz
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

#ifdef ALPAKA_ACC_CPU_BT_OPENACC2_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_CPU_BT_OPENACC2_ENABLED is set, the compiler has to support OpenAcc 2.0 or higher!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/bt/IdxBtRef.hpp>
#include <alpaka/atomic/AtomicStlLock.hpp>
#include <alpaka/atomic/AtomicOpenAcc.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStl.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynRef.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>
#include <alpaka/block/sync/BlockSyncNoOp.hpp>
#include <alpaka/rand/RandStl.hpp>
#include <alpaka/time/TimeStl.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>

#include <boost/core/ignore_unused.hpp>

#include <openacc.h>

#include <limits>
#include <typeinfo>

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOpenAcc2;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU OpenAcc 2.0 accelerator.
        //!
        //! This accelerator allows parallel kernel execution on a CPU device.
        template<
            typename TDim,
            typename TSize>
        class AccCpuOpenAcc2 final :
            public workdiv::WorkDivMembers<TDim, TSize>,
            public idx::gb::IdxGbRef<TDim, TSize>,
            public idx::bt::IdxBtRef<TDim, TSize>,
            public atomic::AtomicHierarchy<
                atomic::AtomicStlLock,  // grid atomics
                atomic::AtomicOpenAcc,  // block atomics
                atomic::AtomicOpenAcc   // thread atomics
            >,
            public math::MathStl,
            public block::shared::dyn::BlockSharedMemDynRef,
            public block::shared::st::BlockSharedMemStMasterSync,
            public block::sync::BlockSyncNoOp,
            public rand::RandStl,
            public time::TimeStl
        {
        public:
            // Partial specialization with the correct TDim and TSize is not allowed.
            template<
                typename TDim2,
                typename TSize2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::exec::ExecCpuOpenAcc2;

        private:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuOpenAcc2(
                TWorkDiv const & workDiv,
                uint8_t * const pBlockSharedMemDyn) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    idx::gb::IdxGbRef<TDim, TSize>(m_gridBlockIdx),
                    idx::bt::IdxBtRef<TDim, TSize>(m_blockThreadIdx),
                    atomic::AtomicHierarchy<
                        atomic::AtomicStlLock,  // atomics between grids
                        atomic::AtomicOpenAcc,  // atomics between blocks
                        atomic::AtomicOpenAcc   // atomics between threads
                    >(),
                    math::MathStl(),
                    block::shared::dyn::BlockSharedMemDynRef(pBlockSharedMemDyn),
                    block::shared::st::BlockSharedMemStMasterSync(
                        [this](){block::sync::syncBlockThreads(*this);},
                        [this](){return (m_blockThreadIdx.sum() == 0u);}),
                    block::sync::BlockSyncNoOp(),
                    rand::RandStl(),
                    time::TimeStl(),
                    m_gridBlockIdx(Vec<TDim, TSize>::zeros()),
                    m_blockThreadIdx(Vec<TDim, TSize>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOpenAcc2(AccCpuOpenAcc2 const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOpenAcc2(AccCpuOpenAcc2 &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOpenAcc2 const &) -> AccCpuOpenAcc2 & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOpenAcc2 &&) -> AccCpuOpenAcc2 & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AccCpuOpenAcc2() = default;

        private:
            // getIdx
            Vec<TDim, TSize> mutable m_gridBlockIdx;    //!< The index of the currently executed block.
            Vec<TDim, TSize> mutable m_blockThreadIdx;  //!< The index of the currently executed thread.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccCpuOpenAcc2<TDim, TSize>>
            {
                using type = acc::AccCpuOpenAcc2<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU OpenAcc 2.0 accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccCpuOpenAcc2<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_CI
                    auto const blockThreadCountMax(static_cast<TSize>(4));
#else
                    auto const blockThreadCountMax(static_cast<TSize>(4/*FIXME: ::omp_get_num_procs()*/));
#endif
                    return {
                        // m_multiProcessorCount
                        static_cast<TSize>(1),
                        // m_gridBlockExtentMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TSize>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TSize>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TSize>::max()};
                }
            };
            //#############################################################################
            //! The CPU OpenAcc 2.0 accelerator name trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccCpuOpenAcc2<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOpenAcc2<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 accelerator device type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccCpuOpenAcc2<TDim, TSize>>
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
            //! The CPU OpenAcc 2.0 accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccCpuOpenAcc2<TDim, TSize>>
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
            //! The CPU OpenAcc 2.0 accelerator executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuOpenAcc2<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 executor platform type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct PltfType<
                acc::AccCpuOpenAcc2<TDim, TSize>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 accelerator size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccCpuOpenAcc2<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
