/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/idx/gb/IdxGbLinear.hpp>
#include <alpaka/idx/bt/IdxBtOmp.hpp>
#include <alpaka/atomic/AtomicOmpBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStOmp5.hpp>
#include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>
#include <alpaka/intrinsic/IntrinsicFallback.hpp>
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
#include <alpaka/dev/DevOmp5.hpp>

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
        class TaskKernelOmp5;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU OpenMP 5.0 accelerator.
        //!
        //! This accelerator allows parallel kernel execution on an OpenMP target device.
        template<
            typename TDim,
            typename TIdx>
        class AccOmp5 final :
            public workdiv::WorkDivMembers<TDim, TIdx>,
            public idx::gb::IdxGbLinear<TDim, TIdx>,
            public idx::bt::IdxBtOmp<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicOmpBuiltIn,   // grid atomics
                atomic::AtomicOmpBuiltIn,    // block atomics
                atomic::AtomicOmpBuiltIn     // thread atomics
            >,
            public math::MathStdLib,
            public block::dyn::BlockSharedMemDynMember<>,
            public block::st::BlockSharedMemStOmp5,
            public block::BlockSyncBarrierOmp,
            // cannot determine which intrinsics are safe to use (depends on target), using fallback
            public intrinsic::IntrinsicFallback,
            public rand::RandStdLib,
            public time::TimeOmp,
            public warp::WarpSingleThread,
            public concepts::Implements<ConceptAcc, AccOmp5<TDim, TIdx>>
        {
            static_assert(sizeof(TIdx) >= sizeof(int), "Index type is not supported, consider using int or a larger type.");
        public:
            // Partial specialization with the correct TDim and TIdx is not allowed.
            template<
                typename TDim2,
                typename TIdx2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::kernel::TaskKernelOmp5;

        private:
            //-----------------------------------------------------------------------------
            AccOmp5(
                Vec<TDim, TIdx> const & gridBlockExtent,
                Vec<TDim, TIdx> const & blockThreadExtent,
                Vec<TDim, TIdx> const & threadElemExtent,
                TIdx const & gridBlockIdx,
                std::size_t const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivMembers<TDim, TIdx>(gridBlockExtent, blockThreadExtent, threadElemExtent),
                    idx::gb::IdxGbLinear<TDim, TIdx>(gridBlockIdx),
                    idx::bt::IdxBtOmp<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicOmpBuiltIn,// atomics between grids
                        atomic::AtomicOmpBuiltIn, // atomics between blocks
                        atomic::AtomicOmpBuiltIn  // atomics between threads
                    >(),
                    math::MathStdLib(),
                    block::dyn::BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes),
                    //! \TODO can with some TMP determine the amount of statically alloced smem from the kernelFuncObj?
                    block::st::BlockSharedMemStOmp5(staticMemBegin(), staticMemCapacity()),
                    block::BlockSyncBarrierOmp(),
                    rand::RandStdLib(),
                    time::TimeOmp()
            {}

        public:
            //-----------------------------------------------------------------------------
            AccOmp5(AccOmp5 const &) = delete;
            //-----------------------------------------------------------------------------
            AccOmp5(AccOmp5 &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccOmp5 const &) -> AccOmp5 & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccOmp5 &&) -> AccOmp5 & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccOmp5() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenMP 5.0 accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccOmp5<TDim, TIdx>>
            {
                using type = acc::AccOmp5<TDim, TIdx>;
            };
            //#############################################################################
            //! The OpenMP 5.0 accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccOmp5<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    DevOmp5 const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    alpaka::ignore_unused(dev);

#if defined(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE) && ALPAKA_OFFLOAD_MAX_BLOCK_SIZE>0
                    auto const blockThreadCount = std::min(::omp_get_max_threads(), ALPAKA_OFFLOAD_MAX_BLOCK_SIZE);
#else
                    auto const blockThreadCount = ::omp_get_max_threads();
#endif
#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(std::min(4, blockThreadCount)));
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(std::min(4, ::omp_get_max_threads())));
#else
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(blockThreadCount));
                    //! \todo for a later OpenMP (or when compilers work with a GPU target): fix max block size for target
                    //!  On CPU we would want
                    //!  gridBlockCountMax = ::omp_get_max_threads() / blockThreadCountMax
                    //!  but this would lead to only one block running on GPU, or too small blocks (see ALPAKA_OFFLOAD_MAX_BLOCK_SIZE).
                    //!  OpenMP 5.0 may actually mandate, that
                    //!  ::omp_get_max_threads() == max_teams * threads_per_team ,
                    //!  however with the maximum grid size (i.e. max_teams) being INT_MAX this may not work.
                    //!  We actually want to set
                    //!  gridBlockCountMax = ::omp_get_max_teams()
                    //!  but there is no function ::omp_get_max_teams().
                    //!  Instead we set ::omp_get_max_threads() again, to have a
                    //!  number which does not kill CPUs and is reasonable
                    //!  (::omp_get_max_threads() seems to return the block size)
                    //!  for GPUs.
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(::omp_get_max_threads()));
#endif
                    return {
                        // m_multiProcessorCount
                        static_cast<TIdx>(gridBlockCountMax),
                        // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        acc::AccOmp5<TDim, TIdx>::staticAllocBytes()};
                }
            };
            //#############################################################################
            //! The OpenMP 5.0 accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccOmp5<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccOmp5<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace traits
    {
        //#############################################################################
        //! The OpenMP 5.0 accelerator device type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct DevType<
            acc::AccOmp5<TDim, TIdx>>
        {
            using type = DevOmp5;
        };
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenMP 5.0 accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccOmp5<TDim, TIdx>>
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
            //! The OpenMP 5.0 accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccOmp5<TDim, TIdx>,
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
                        kernel::TaskKernelOmp5<
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
    namespace traits
    {
        //#############################################################################
        //! The OpenMP 5.0 execution task platform type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct PltfType<
            acc::AccOmp5<TDim, TIdx>>
        {
            using type = PltfOmp5;
        };
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenMP 5.0 accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccOmp5<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
