/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

// Base classes.
#include <alpaka/ctx/block/CtxBlockOacc.hpp>
#include <alpaka/acc/AccOacc.hpp>

// Specialized traits.
#include <alpaka/idx/Traits.hpp>
#include <alpaka/block/shared/dyn/Traits.hpp>
#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/block/sync/Traits.hpp>

#include <limits>
#include <typeinfo>

namespace alpaka
{
    //#############################################################################
    //! The OpenACC block context.
    template<
        typename TDim,
        typename TIdx>
    class CtxThreadOacc final :
        public AccOacc<TDim, TIdx>,
        public concepts::Implements<ConceptWorkDiv, CtxThreadOacc<TDim, TIdx>>,
        public concepts::Implements<ConceptBlockSharedDyn, CtxThreadOacc<TDim, TIdx>>,
        public concepts::Implements<ConceptBlockSharedSt, CtxThreadOacc<TDim, TIdx>>,
        public concepts::Implements<ConceptBlockSync, CtxThreadOacc<TDim, TIdx>>,
        public concepts::Implements<ConceptIdxGb, CtxThreadOacc<TDim, TIdx>>
    {
    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<
            typename TDim2,
            typename TIdx2,
            typename TKernelFnObj,
            typename... TArgs>
        friend class ::alpaka::TaskKernelOacc;

    private:
        //-----------------------------------------------------------------------------
        CtxThreadOacc(
            TIdx const & blockThreadIdx,
            CtxBlockOacc<TDim, TIdx>& blockShared) :
                AccOacc<TDim, TIdx>(blockThreadIdx),
                m_blockShared(blockShared)
        {}

    public:
        //-----------------------------------------------------------------------------
        CtxThreadOacc(CtxThreadOacc const &) = delete;
        //-----------------------------------------------------------------------------
        CtxThreadOacc(CtxThreadOacc &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(CtxThreadOacc const &) -> CtxThreadOacc & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(CtxThreadOacc &&) -> CtxThreadOacc & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~CtxThreadOacc() = default;

        CtxBlockOacc<TDim, TIdx>& m_blockShared;
    };

    namespace traits
    {
        //#############################################################################
        //! The GPU CUDA accelerator grid block index get trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetIdx<
            CtxThreadOacc<TDim, TIdx>,
            origin::Grid,
            unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current block in the grid.
            template<
                typename TWorkDiv>
            static auto getIdx(
                CtxThreadOacc<TDim, TIdx> const & idx,
                TWorkDiv const & workDiv)
            -> Vec<TDim, TIdx>
            {
                // // \TODO: Would it be faster to precompute the index and cache it inside an array?
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(idx.m_blockShared.m_gridBlockIdx),
                    getWorkDiv<Grid, Blocks>(workDiv));
            }
        };

        template<
            typename TIdx>
        struct GetIdx<
            CtxThreadOacc<DimInt<1u>, TIdx>,
            origin::Grid,
            unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current block in the grid.
            template<
                typename TWorkDiv>
            static auto getIdx(
                CtxThreadOacc<DimInt<1u>, TIdx> const & idx,
                TWorkDiv const &)
            -> Vec<DimInt<1u>, TIdx>
            {
                return idx.m_blockShared.m_gridBlockIdx;
            }
        };

        //#############################################################################
        template<
            typename T,
            typename TDim,
            typename TIdx>
        struct GetMem<
            T,
            CtxThreadOacc<TDim, TIdx>>
        {
#if BOOST_COMP_GNUC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
            //-----------------------------------------------------------------------------
            static auto getMem(
                CtxThreadOacc<TDim, TIdx> const &mem)
            -> T *
            {
                return reinterpret_cast<T*>(mem.m_blockShared.dynMemBegin());
            }
#if BOOST_COMP_GNUC
#pragma GCC diagnostic pop
#endif
        };

        //#############################################################################
        template<
            typename T,
            typename TDim,
            typename TIdx,
            std::size_t TuniqueId>
        struct AllocVar<
            T,
            TuniqueId,
            CtxThreadOacc<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            static auto allocVar(
                CtxThreadOacc<TDim, TIdx> const &smem)
            -> T &
            {
                return alpaka::allocVar<T, TuniqueId>(smem.m_blockShared);
            }
        };

        //#############################################################################
        template<
            typename TDim,
            typename TIdx>
        struct FreeMem<
            CtxThreadOacc<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            static auto freeMem(
                CtxThreadOacc<TDim, TIdx> const & smem)
            -> void
            {
                alpaka::freeMem(smem.m_blockShared);
            }
        };

        //#############################################################################
        template<
            typename TDim,
            typename TIdx>
        struct SyncBlockThreads<
            CtxThreadOacc<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            //! Execute op with single thread (any idx, last thread to
            //! arrive at barrier executes) syncing before and after
            template<
                typename TOp>
            ALPAKA_FN_HOST static auto masterOpBlockThreads(
                CtxThreadOacc<TDim, TIdx> const & acc,
                TOp &&op)
            -> void
            {
                SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>::masterOpBlockThreads(acc.m_blockShared, op);
            }

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto syncBlockThreads(
                CtxThreadOacc<TDim, TIdx> const & acc)
            -> void
            {
                SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>::syncBlockThreads(acc.m_blockShared);
            }
        };

        //#############################################################################
        template<
            typename TOp,
            typename TDim,
            typename TIdx>
        struct SyncBlockThreadsPredicate<
            TOp,
            CtxThreadOacc<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                CtxThreadOacc<TDim, TIdx> const & acc,
                int predicate)
            -> int
            {
                return SyncBlockThreadsPredicate<TOp, CtxBlockOacc<TDim, TIdx>>::syncBlockThreadsPredicate(
                        acc.m_blockShared,
                        predicate
                    );
            }
        };

        //#############################################################################
        //! The CPU OpenACC accelerator device type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct DevType<
            CtxThreadOacc<TDim, TIdx>>
        {
            using type = DevOacc;
        };

        //#############################################################################
        //! The OpenACC accelerator dimension getter trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct DimType<
            CtxThreadOacc<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The OpenACC accelerator execution task type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<
            CtxThreadOacc<TDim, TIdx>,
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
                    TaskKernelOacc<
                        TDim,
                        TIdx,
                        TKernelFnObj,
                        TArgs...>(
                            workDiv,
                            kernelFnObj,
                            std::forward<TArgs>(args)...);
            }
        };

        //#############################################################################
        //! The CPU OpenACC execution task platform type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct PltfType<
            CtxThreadOacc<TDim, TIdx>>
        {
            using type = PltfOacc;
        };

        //#############################################################################
        //! The CPU OpenACC accelerator idx type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct IdxType<
            CtxThreadOacc<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //#############################################################################
        //! The CtxThreadsOacc grid block extent trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetWorkDiv<
            CtxThreadOacc<TDim, TIdx>,
            origin::Grid,
            unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in each dimension of the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                CtxThreadOacc<TDim, TIdx> const & workDiv)
            -> Vec<TDim, TIdx>
            {
                return GetWorkDiv<
                    WorkDivMembers<TDim, TIdx>,
                    origin::Grid,
                    unit::Blocks>::getWorkDiv(workDiv.m_blockShared);
            }
        };

        //#############################################################################
        //! The CtxThreadOacc block thread extent trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetWorkDiv<
            CtxThreadOacc<TDim, TIdx>,
            origin::Block,
            unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of threads in each dimension of a block.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                CtxThreadOacc<TDim, TIdx> const & workDiv)
            -> Vec<TDim, TIdx>
            {
                return GetWorkDiv<
                    WorkDivMembers<TDim, TIdx>,
                    origin::Block,
                    unit::Threads>::getWorkDiv(workDiv.m_blockShared);
            }
        };

        //#############################################################################
        //! The CtxThreadOacc thread element extent trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetWorkDiv<
            CtxThreadOacc<TDim, TIdx>,
            origin::Thread,
            unit::Elems>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of elements in each dimension of a thread.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                CtxThreadOacc<TDim, TIdx> const & workDiv)
            -> Vec<TDim, TIdx>
            {
                return GetWorkDiv<
                    WorkDivMembers<TDim, TIdx>,
                    origin::Thread,
                    unit::Elems>::getWorkDiv(workDiv.m_blockShared);
            }
        };
    }
}

#endif
