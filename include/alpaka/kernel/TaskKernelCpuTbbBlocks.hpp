/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/acc/AccCpuTbbBlocks.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/meta/ApplyTuple.hpp>
#    include <alpaka/meta/NdLoop.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <functional>
#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

#    include <tbb/blocked_range.h>
#    include <tbb/parallel_for.h>
#    include <tbb/task_group.h>

namespace alpaka
{
    //! The CPU TBB block accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuTbbBlocks final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelCpuTbbBlocks(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }
        TaskKernelCpuTbbBlocks(TaskKernelCpuTbbBlocks const&) = default;
        TaskKernelCpuTbbBlocks(TaskKernelCpuTbbBlocks&&) = default;
        auto operator=(TaskKernelCpuTbbBlocks const&) -> TaskKernelCpuTbbBlocks& = default;
        auto operator=(TaskKernelCpuTbbBlocks&&) -> TaskKernelCpuTbbBlocks& = default;
        ~TaskKernelCpuTbbBlocks() = default;

        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()() const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent(getWorkDiv<Grid, Blocks>(*this));
            auto const blockThreadExtent(getWorkDiv<Block, Threads>(*this));
            auto const threadElemExtent(getWorkDiv<Thread, Elems>(*this));

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes(meta::apply(
                [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return getBlockSharedMemDynSizeBytes<AccCpuTbbBlocks<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif
            // Bind all arguments except the accelerator.
            // TODO: With C++14 we could create a perfectly argument forwarding function object within the constructor.
            auto const boundKernelFnObj(meta::apply(
                [this](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return std::bind(std::ref(m_kernelFnObj), std::placeholders::_1, std::ref(args)...);
                },
                m_args));

            // The number of blocks in the grid.
            TIdx const numBlocksInGrid(gridBlockExtent.prod());

            if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
            {
                throw std::runtime_error("A block for the TBB accelerator can only ever have one single thread!");
            }

            tbb::parallel_for(static_cast<TIdx>(0), static_cast<TIdx>(numBlocksInGrid), [&](TIdx i) {
                AccCpuTbbBlocks<TDim, TIdx> acc(
                    *static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
                    blockSharedMemDynSizeBytes);

                acc.m_gridBlockIdx = mapIdx<TDim::value>(Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(i)), gridBlockExtent);

                boundKernelFnObj(acc);

                freeSharedVars(acc);
            });
        }

    private:
        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace traits
    {
        //! The CPU TBB block execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccCpuTbbBlocks<TDim, TIdx>;
        };

        //! The CPU TBB block execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevCpu;
        };

        //! The CPU TBB block execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The CPU TBB block execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfCpu;
        };

        //! The CPU TBB block execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
