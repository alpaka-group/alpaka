/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

//#############################################################################
// Define the version number.
//#############################################################################

#include <boost/predef/version_number.h>    // BOOST_VERSION_NUMBER

//! The alpaka library version number
#define ALPAKA_VERSION BOOST_VERSION_NUMBER(1, 1, 0)

//#############################################################################
// Include the whole library.
//#############################################################################

//-----------------------------------------------------------------------------
// acc
//-----------------------------------------------------------------------------
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    #include <alpaka/acc/AccCpuSerial.hpp>
    #include <alpaka/exec/ExecCpuSerial.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    #include <alpaka/acc/AccCpuThreads.hpp>
    #include <alpaka/exec/ExecCpuThreads.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
    #include <alpaka/acc/AccCpuFibers.hpp>
    #include <alpaka/exec/ExecCpuFibers.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    #if _OPENMP < 200203
        #error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
    #endif
    #include <alpaka/acc/AccCpuOmp2Blocks.hpp>
    #include <alpaka/exec/ExecCpuOmp2Blocks.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    #if _OPENMP < 200203
        #error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
    #endif
    #include <alpaka/acc/AccCpuOmp2Threads.hpp>
    #include <alpaka/exec/ExecCpuOmp2Threads.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    #if _OPENMP < 201307
        #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
    #endif
    #include <alpaka/acc/AccCpuOmp4.hpp>
    #include <alpaka/exec/ExecCpuOmp4.hpp>
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/acc/AccGpuCudaRt.hpp>
    #include <alpaka/exec/ExecGpuCudaRt.hpp>
#endif
#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/Traits.hpp>

//-----------------------------------------------------------------------------
// atomic
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/atomic/AtomicCudaBuiltIn.hpp>
#endif
#include <alpaka/atomic/AtomicNoOp.hpp>
#ifdef _OPENMP
    #include <alpaka/atomic/AtomicOmpCritSec.hpp>
#endif
#include <alpaka/atomic/AtomicStlLock.hpp>
#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>

//-----------------------------------------------------------------------------
// block
//-----------------------------------------------------------------------------
    //-----------------------------------------------------------------------------
    // shared
    //-----------------------------------------------------------------------------
    #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
        #include <alpaka/block/shared/BlockSharedAllocCudaBuiltIn.hpp>
    #endif
    #include <alpaka/block/shared/BlockSharedAllocMasterSync.hpp>
    #include <alpaka/block/shared/BlockSharedAllocNoSync.hpp>
    #include <alpaka/block/shared/Traits.hpp>

    //-----------------------------------------------------------------------------
    // sync
    //-----------------------------------------------------------------------------
    #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
        #include <alpaka/block/sync/BlockSyncCudaBuiltIn.hpp>
    #endif
    #ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
        #include <alpaka/block/sync/BlockSyncFiberIdMapBarrier.hpp>
    #endif
    #include <alpaka/block/sync/BlockSyncNoOp.hpp>
    #ifdef _OPENMP
        #include <alpaka/block/sync/BlockSyncOmpBarrier.hpp>
    #endif
    #ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
        #include <alpaka/block/sync/BlockSyncThreadIdMapBarrier.hpp>
    #endif
    #include <alpaka/block/sync/Traits.hpp>

//-----------------------------------------------------------------------------
// dev
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/dev/DevCudaRt.hpp>
#endif
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/Traits.hpp>

//-----------------------------------------------------------------------------
// dim
//-----------------------------------------------------------------------------
#include <alpaka/dim/DimArithmetic.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/dim/Traits.hpp>

//-----------------------------------------------------------------------------
// event
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/event/EventCudaRt.hpp>
#endif
#include <alpaka/event/EventCpu.hpp>
#include <alpaka/event/Traits.hpp>

//-----------------------------------------------------------------------------
// exec
//-----------------------------------------------------------------------------
#include <alpaka/exec/Traits.hpp>

//-----------------------------------------------------------------------------
// extent
//-----------------------------------------------------------------------------
#include <alpaka/extent/Traits.hpp>

//-----------------------------------------------------------------------------
// idx
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/idx/bt/IdxBtCudaBuiltIn.hpp>
#endif
#ifdef _OPENMP
    #include <alpaka/idx/bt/IdxBtOmp.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
    #include <alpaka/idx/bt/IdxBtRefFiberIdMap.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    #include <alpaka/idx/bt/IdxBtRefThreadIdMap.hpp>
#endif
#include <alpaka/idx/bt/IdxBtZero.hpp>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/idx/gb/IdxGbCudaBuiltIn.hpp>
#endif
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/Traits.hpp>

//-----------------------------------------------------------------------------
// kernel
//-----------------------------------------------------------------------------
#include <alpaka/kernel/Traits.hpp>

//-----------------------------------------------------------------------------
// math
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/math/MathCudaBuiltIn.hpp>
#endif
#include <alpaka/math/MathStl.hpp>

//-----------------------------------------------------------------------------
// mem
//-----------------------------------------------------------------------------
#include <alpaka/mem/alloc/AllocCpuBoostAligned.hpp>
#include <alpaka/mem/alloc/AllocCpuNew.hpp>
#include <alpaka/mem/alloc/Traits.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/mem/buf/BufCudaRt.hpp>
#endif
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/buf/BufPlainPtrWrapper.hpp>
#include <alpaka/mem/buf/BufStdContainers.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <alpaka/mem/view/ViewBasic.hpp>
#include <alpaka/mem/view/Traits.hpp>

//-----------------------------------------------------------------------------
// offset
//-----------------------------------------------------------------------------
#include <alpaka/offset/Traits.hpp>

//-----------------------------------------------------------------------------
// rand
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/rand/RandCuRand.hpp>
#endif
#include <alpaka/rand/RandStl.hpp>
#include <alpaka/rand/Traits.hpp>

//-----------------------------------------------------------------------------
// size
//-----------------------------------------------------------------------------
#include <alpaka/size/Traits.hpp>

//-----------------------------------------------------------------------------
// stream
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/stream/StreamCudaRtAsync.hpp>
    #include <alpaka/stream/StreamCudaRtSync.hpp>
#endif
#include <alpaka/stream/StreamCpuAsync.hpp>
#include <alpaka/stream/StreamCpuSync.hpp>
#include <alpaka/stream/Traits.hpp>

//-----------------------------------------------------------------------------
// wait
//-----------------------------------------------------------------------------
#include <alpaka/wait/Traits.hpp>

//-----------------------------------------------------------------------------
// workdiv
//-----------------------------------------------------------------------------
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

//-----------------------------------------------------------------------------
// vec
//-----------------------------------------------------------------------------
#include <alpaka/vec/Vec.hpp>
#include <alpaka/vec/Traits.hpp>

//-----------------------------------------------------------------------------
// core
//-----------------------------------------------------------------------------
#include <alpaka/core/Align.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Fold.hpp>
#include <alpaka/core/ForEachType.hpp>
#include <alpaka/core/MapIdx.hpp>
#include <alpaka/core/NdLoop.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unroll.hpp>
#include <alpaka/core/Vectorize.hpp>
