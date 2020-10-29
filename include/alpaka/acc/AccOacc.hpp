/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
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
#include <alpaka/idx/gb/IdxGbOaccBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtOaccBuiltIn.hpp>
#include <alpaka/atomic/AtomicOaccBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/intrinsic/IntrinsicFallback.hpp>
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/rand/RandStdLib.hpp>
#include <alpaka/time/TimeStdLib.hpp>
#include <alpaka/ctx/block/CtxBlockOacc.hpp>
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
#include <alpaka/dev/DevOacc.hpp>

#include <limits>
#include <typeinfo>

namespace alpaka
{
    template<
        typename TDim,
        typename TIdx,
        typename TKernelFnObj,
        typename... TArgs>
    class TaskKernelOacc;

    // define max gang/worker num because there is no standart way in OpenACC to
    // get this information
#ifndef ALPAKA_OACC_MAX_GANG_NUM
    constexpr size_t oaccMaxGangNum = std::numeric_limits<unsigned int>::max();
#else
    constexpr size_t oaccMaxGangNum = ALPAKA_OACC_MAX_GANG_NUM;
#endif
#if defined(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE) && ALPAKA_OFFLOAD_MAX_BLOCK_SIZE>0
    constexpr size_t oaccMaxWorkerNum = ALPAKA_OFFLOAD_MAX_BLOCK_SIZE;
#else
    constexpr size_t oaccMaxWorkerNum = 1;
#endif

    //#############################################################################
    //! The OpenACC accelerator.
    template<
        typename TDim,
        typename TIdx>
    class AccOacc :
        public gb::IdxGbOaccBuiltIn<TDim, TIdx>, // dummy
        public bt::IdxBtOaccBuiltIn<TDim, TIdx>,
        public AtomicHierarchy<
            AtomicOaccBuiltIn,    // grid atomics
            AtomicOaccBuiltIn,    // block atomics
            AtomicOaccBuiltIn     // thread atomics
        >,
        public math::MathStdLib,
        public rand::RandStdLib,
        public TimeStdLib,
        public warp::WarpSingleThread,
        // NVHPC calls a builtin in the STL implementation, which fails in OpenACC offload, using fallback
        public IntrinsicFallback,
        public concepts::Implements<ConceptAcc, AccOacc<TDim, TIdx>>
    {

    protected:
        //-----------------------------------------------------------------------------
        AccOacc(
            TIdx const & blockThreadIdx) :
                gb::IdxGbOaccBuiltIn<TDim, TIdx>(),
                bt::IdxBtOaccBuiltIn<TDim, TIdx>(blockThreadIdx),
                AtomicHierarchy<
                    AtomicOaccBuiltIn,    // grid atomics
                    AtomicOaccBuiltIn,    // block atomics
                    AtomicOaccBuiltIn     // thread atomics
                >(),
                math::MathStdLib(),
                rand::RandStdLib(),
                TimeStdLib()
        {}

    public:
        //-----------------------------------------------------------------------------
        AccOacc(AccOacc const &) = delete;
        //-----------------------------------------------------------------------------
        AccOacc(AccOacc &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccOacc const &) -> AccOacc & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccOacc &&) -> AccOacc & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AccOacc() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU OpenACC accelerator accelerator type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct AccType<
            AccOacc<TDim, TIdx>>
        {
            using type = AccOacc<TDim, TIdx>;
        };
        //#############################################################################
        //! The CPU OpenACC accelerator device properties get trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetAccDevProps<
            AccOacc<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(
                DevOacc const & dev)
            -> AccDevProps<TDim, TIdx>
            {
                alpaka::ignore_unused(dev);

#ifdef ALPAKA_CI
                auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(std::min(static_cast<size_t>(2u), oaccMaxWorkerNum)));
                auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(std::min(static_cast<size_t>(2u), oaccMaxGangNum)));
#else
                auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(oaccMaxWorkerNum));
                auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(oaccMaxGangNum));
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
                    CtxBlockOacc<TDim, TIdx>::staticAllocBytes()};
            }
        };
        //#############################################################################
        //! The OpenACC accelerator name trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetAccName<
            AccOacc<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName()
            -> std::string
            {
                return "AccOacc<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The CPU OpenACC accelerator device type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct DevType<
            AccOacc<TDim, TIdx>>
        {
            using type = DevOacc;
        };

        //#############################################################################
        //! The CPU OpenACC accelerator dimension getter trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct DimType<
            AccOacc<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU OpenACC accelerator execution task type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<
            AccOacc<TDim, TIdx>,
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
            AccOacc<TDim, TIdx>>
        {
            using type = PltfOacc;
        };

        //#############################################################################
        //! The CPU OpenACC accelerator idx type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct IdxType<
            AccOacc<TDim, TIdx>>
        {
            using type = TIdx;
        };
    }
}

#endif
