/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivOmp4BuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbOmp4BuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtOmp4BuiltIn.hpp>
#include <alpaka/atomic/AtomicOmpBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynOmp4.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStOmp4.hpp>
#include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>
#include <alpaka/intrinsic/IntrinsicCpu.hpp>
#include <alpaka/rand/RandStdLib.hpp>
#include <alpaka/time/TimeOmp.hpp>
#include <alpaka/warp/WarpSingleThread.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevOmp4.hpp>

#include <limits>
#include <typeinfo>

namespace alpaka
{
    namespace kernel
    {
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelCpuOmp4;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU OpenMP 4.0 accelerator.
        //!
        //! This accelerator allows parallel kernel execution on a CPU device.
        //! It uses CPU OpenMP4 to implement the parallelism.
        template<
            typename TDim,
            typename TIdx>
        class AccCpuOmp4 final :
            public workdiv::WorkDivOmp4BuiltIn<TDim, TIdx>,
            public idx::gb::IdxGbOmp4BuiltIn<TDim, TIdx>,
            public idx::bt::IdxBtOmp4BuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicOmpBuiltIn,   // grid atomics
                atomic::AtomicOmpBuiltIn,    // block atomics
                atomic::AtomicOmpBuiltIn     // thread atomics
            >,
            public math::MathStdLib,
            public block::shared::dyn::BlockSharedMemDynOmp4,
            public block::shared::st::BlockSharedMemStOmp4,
            public block::sync::BlockSyncBarrierOmp,
            public intrinsic::IntrinsicCpu,
            public rand::RandStdLib,
            public time::TimeOmp,
            public warp::WarpSingleThread,
            public concepts::Implements<ConceptAcc, AccCpuOmp4<TDim, TIdx>>
        {
        public:
            // Partial specialization with the correct TDim and TIdx is not allowed.
            template<
                typename TDim2,
                typename TIdx2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::kernel::TaskKernelCpuOmp4;

        private:
            //-----------------------------------------------------------------------------
            AccCpuOmp4(
                vec::Vec<TDim, TIdx> const & gridBlockExtent,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                TIdx const & teamOffset,
                TIdx const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivOmp4BuiltIn<TDim, TIdx>(threadElemExtent, blockThreadExtent, gridBlockExtent),
                    idx::gb::IdxGbOmp4BuiltIn<TDim, TIdx>(teamOffset),
                    idx::bt::IdxBtOmp4BuiltIn<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicOmpBuiltIn,// atomics between grids
                        atomic::AtomicOmpBuiltIn, // atomics between blocks
                        atomic::AtomicOmpBuiltIn  // atomics between threads
                    >(),
                    math::MathStdLib(),
                    block::shared::dyn::BlockSharedMemDynOmp4(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    //! \TODO can with some TMP determine the amount of statically alloced smem from the kernelFuncObj?
                    block::shared::st::BlockSharedMemStOmp4(staticMemBegin()),
                    block::sync::BlockSyncBarrierOmp(),
                    rand::RandStdLib(),
                    time::TimeOmp(),
                    m_gridBlockIdx(vec::Vec<TDim, TIdx>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            AccCpuOmp4(AccCpuOmp4 const &) = delete;
            //-----------------------------------------------------------------------------
            AccCpuOmp4(AccCpuOmp4 &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccCpuOmp4 const &) -> AccCpuOmp4 & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccCpuOmp4 &&) -> AccCpuOmp4 & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccCpuOmp4() = default;

        private:
            // getIdx
            vec::Vec<TDim, TIdx> m_gridBlockIdx;    //!< The index of the currently executed block.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccCpuOmp4<TDim, TIdx>>
            {
                using type = acc::AccCpuOmp4<TDim, TIdx>;
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccCpuOmp4<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevOmp4 const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    alpaka::ignore_unused(dev);

#if defined(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE) && ALPAKA_OFFLOAD_MAX_BLOCK_SIZE>0
                    auto const blockThreadCount = ALPAKA_OFFLOAD_MAX_BLOCK_SIZE;
#else
                    auto const blockThreadCount = ::omp_get_max_threads();
#endif
#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(std::min(4, blockThreadCount)));
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(std::min(4, ::omp_get_max_threads())));
#else
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(blockThreadCount));
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(::omp_get_max_threads())); //! \todo fix max block size for target
#endif
                    return {
                        // m_multiProcessorCount
                        static_cast<TIdx>(gridBlockCountMax),
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
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        dev::getMemBytes( dev )};
                }
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccCpuOmp4<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp4<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccCpuOmp4<TDim, TIdx>>
            {
                using type = dev::DevOmp4;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccCpuOmp4<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccCpuOmp4<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs && ... args)
                {
                    return
                        kernel::TaskKernelCpuOmp4<
                            TDim,
                            TIdx,
                            TKernelFnObj,
                            TArgs...>(
                                workDiv,
                                kernelFnObj,
                                std::forward<TArgs>(args)...);
                }
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccCpuOmp4<TDim, TIdx>>
            {
                using type = pltf::PltfOmp4;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccCpuOmp4<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
