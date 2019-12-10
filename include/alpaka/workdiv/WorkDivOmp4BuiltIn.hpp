/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
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

#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Omp4.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU CUDA accelerator work division.
        template<
            typename TDim,
            typename TIdx>
        class WorkDivOmp4BuiltIn : public concepts::Implements<ConceptWorkDiv, WorkDivOmp4BuiltIn<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            WorkDivOmp4BuiltIn(
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                vec::Vec<TDim, TIdx> const & gridBlockExtent) :
                    m_threadElemExtent(threadElemExtent),
                    m_blockThreadExtent(blockThreadExtent),
                    m_gridBlockExtent(gridBlockExtent)
            {
                // printf("WorkDivOmp4BuiltIn ctor threadElemExtent %d\n", int(threadElemExtent[0]));
            }
            //-----------------------------------------------------------------------------
            WorkDivOmp4BuiltIn(WorkDivOmp4BuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            WorkDivOmp4BuiltIn(WorkDivOmp4BuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivOmp4BuiltIn const &) -> WorkDivOmp4BuiltIn & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivOmp4BuiltIn &&) -> WorkDivOmp4BuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~WorkDivOmp4BuiltIn() = default;

        public:
            // \TODO: Optimize! Add WorkDivCudaBuiltInNoElems that has no member m_threadElemExtent as well as AccGpuCudaRtNoElems.
            // Use it instead of AccGpuCudaRt if the thread element extent is one to reduce the register usage.
            vec::Vec<TDim, TIdx> const m_threadElemExtent, m_blockThreadExtent, m_gridBlockExtent;
        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                workdiv::WorkDivOmp4BuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                workdiv::WorkDivOmp4BuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division grid block extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivOmp4BuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                static auto getWorkDiv(
                    WorkDivOmp4BuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_gridBlockExtent;
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division block thread extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivOmp4BuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                static auto getWorkDiv(
                    WorkDivOmp4BuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_blockThreadExtent;
                    // return vec::Vec<TDim, TIdx>(static_cast<TIdx>(omp_get_num_threads()));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division thread element extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivOmp4BuiltIn<TDim, TIdx>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                static auto getWorkDiv(
                    WorkDivOmp4BuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
