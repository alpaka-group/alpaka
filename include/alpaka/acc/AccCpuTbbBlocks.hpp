/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

// Base classes.
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/atomic/AtomicNoOp.hpp>
#    include <alpaka/atomic/AtomicStdLibLock.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMember.hpp>
#    include <alpaka/block/sync/BlockSyncNoOp.hpp>
#    include <alpaka/idx/bt/IdxBtZero.hpp>
#    include <alpaka/idx/gb/IdxGbRef.hpp>
#    include <alpaka/intrinsic/IntrinsicCpu.hpp>
#    include <alpaka/math/MathStdLib.hpp>
#    include <alpaka/rand/RandStdLib.hpp>
#    include <alpaka/time/TimeStdLib.hpp>
#    include <alpaka/warp/WarpSingleThread.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/dev/DevCpu.hpp>

#    include <memory>
#    include <typeinfo>

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuTbbBlocks;

    //#############################################################################
    //! The CPU TBB block accelerator.
    template<
        typename TDim,
        typename TIdx>
    class AccCpuTbbBlocks final :
        public WorkDivMembers<TDim, TIdx>,
        public gb::IdxGbRef<TDim, TIdx>,
        public bt::IdxBtZero<TDim, TIdx>,
        public AtomicHierarchy<
            AtomicStdLibLock<16>, // grid atomics
            AtomicStdLibLock<16>, // block atomics
            AtomicNoOp         // thread atomics
        >,
        public math::MathStdLib,
        public BlockSharedMemDynMember<>,
        public BlockSharedMemStMember<>,
        public BlockSyncNoOp,
        public IntrinsicCpu,
        public rand::RandStdLib,
        public TimeStdLib,
        public warp::WarpSingleThread,
        public concepts::Implements<ConceptAcc, AccCpuTbbBlocks<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelCpuTbbBlocks;

    private:
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST AccCpuTbbBlocks(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , bt::IdxBtZero<TDim, TIdx>()
            , AtomicHierarchy<
                  AtomicStdLibLock<16>, // atomics between grids
                  AtomicStdLibLock<16>, // atomics between blocks
                  AtomicNoOp // atomics between threads
                  >()
            , math::MathStdLib()
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMember<>(staticMemBegin(), staticMemCapacity())
            , BlockSyncNoOp()
            , rand::RandStdLib()
            , TimeStdLib()
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AccCpuTbbBlocks(AccCpuTbbBlocks const&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AccCpuTbbBlocks(AccCpuTbbBlocks&&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AccCpuTbbBlocks const&) -> AccCpuTbbBlocks& = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AccCpuTbbBlocks&&) -> AccCpuTbbBlocks& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AccCpuTbbBlocks() = default;

    private:
        // getIdx
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU TBB block accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = AccCpuTbbBlocks<TDim, TIdx>;
        };
        //#############################################################################
        //! The CPU TBB block accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuTbbBlocks<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> AccDevProps<TDim, TIdx>
            {
                alpaka::ignore_unused(dev);

                return {// m_multiProcessorCount
                        static_cast<TIdx>(1),
                        // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::ones(),
                        // m_blockThreadCountMax
                        static_cast<TIdx>(1),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(AccCpuTbbBlocks<TDim, TIdx>::staticAllocBytes())};
            }
        };
        //#############################################################################
        //! The CPU TBB block accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuTbbBlocks<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccCpuTbbBlocks<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The CPU TBB block accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The CPU TBB block accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU TBB block accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuTbbBlocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //#############################################################################
        //! The CPU TBB block execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = PltfCpu;
        };

        //#############################################################################
        //! The CPU TBB block accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
