/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

// Base classes.
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/atomic/AtomicNoOp.hpp>
#    include <alpaka/atomic/AtomicOmpBuiltIn.hpp>
#    include <alpaka/atomic/AtomicStdLibLock.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMember.hpp>
#    include <alpaka/block/sync/BlockSyncNoOp.hpp>
#    include <alpaka/idx/bt/IdxBtZero.hpp>
#    include <alpaka/idx/gb/IdxGbRef.hpp>
#    include <alpaka/intrinsic/IntrinsicCpu.hpp>
#    include <alpaka/math/MathStdLib.hpp>
#    include <alpaka/rand/RandStdLib.hpp>
#    include <alpaka/time/TimeOmp.hpp>
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

#    include <limits>
#    include <typeinfo>

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuOmp2Blocks;

    //#############################################################################
    //! The CPU OpenMP 2.0 block accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a CPU device.
    //! It uses OpenMP 2.0 to implement the grid block parallelism.
    //! The block idx is restricted to 1x1x1.
    template<
        typename TDim,
        typename TIdx>
    class AccCpuOmp2Blocks final :
        public WorkDivMembers<TDim, TIdx>,
        public gb::IdxGbRef<TDim, TIdx>,
        public bt::IdxBtZero<TDim, TIdx>,
        public AtomicHierarchy<
            AtomicStdLibLock<16>,   // grid atomics
            AtomicOmpBuiltIn,    // block atomics
            AtomicNoOp           // thread atomics
        >,
        public math::MathStdLib,
        public BlockSharedMemDynMember<>,
        public BlockSharedMemStMember<>,
        public BlockSyncNoOp,
        public IntrinsicCpu,
        public rand::RandStdLib,
        public TimeOmp,
        public warp::WarpSingleThread,
        public concepts::Implements<ConceptAcc, AccCpuOmp2Blocks<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelCpuOmp2Blocks;

    private:
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST AccCpuOmp2Blocks(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , bt::IdxBtZero<TDim, TIdx>()
            , AtomicHierarchy<
                  AtomicStdLibLock<16>, // atomics between grids
                  AtomicOmpBuiltIn, // atomics between blocks
                  AtomicNoOp // atomics between threads
                  >()
            , math::MathStdLib()
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMember<>(staticMemBegin(), staticMemCapacity())
            , BlockSyncNoOp()
            , rand::RandStdLib()
            , TimeOmp()
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AccCpuOmp2Blocks(AccCpuOmp2Blocks const&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AccCpuOmp2Blocks(AccCpuOmp2Blocks&&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AccCpuOmp2Blocks const&) -> AccCpuOmp2Blocks& = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AccCpuOmp2Blocks&&) -> AccCpuOmp2Blocks& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AccCpuOmp2Blocks() = default;

    private:
        // getIdx
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = AccCpuOmp2Blocks<TDim, TIdx>;
        };
        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> alpaka::AccDevProps<TDim, TIdx>
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
                        static_cast<size_t>(AccCpuOmp2Blocks<TDim, TIdx>::staticAllocBytes())};
            }
        };
        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccCpuOmp2Blocks<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuOmp2Blocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 block execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = PltfCpu;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
