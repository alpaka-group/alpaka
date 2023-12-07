/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Antonio Di Pilato <tony.dipilato03@gmail.com>
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Luca Ferragina <luca.ferragina@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
 * SPDX-FileContributor: Jakob Krude <jakob.krude@hotmail.com>
 * SPDX-FileContributor: Felice Pantaleo <felice.pantaleo@cern.ch>
 * SPDX-FileContributor: Aurora Perego <aurora.perego@cern.ch>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Jiří Vyskočil <jiri@vyskocil.com>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 * SPDX-FileContributor: Erik Zenker <erikzenker@posteo.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Include the whole library.

// version number
#include "alpaka/version.hpp"
// acc
#include "alpaka/acc/AccCpuOmp2Blocks.hpp"
#include "alpaka/acc/AccCpuOmp2Threads.hpp"
#include "alpaka/acc/AccCpuSerial.hpp"
#include "alpaka/acc/AccCpuSycl.hpp"
#include "alpaka/acc/AccCpuTbbBlocks.hpp"
#include "alpaka/acc/AccCpuThreads.hpp"
#include "alpaka/acc/AccDevProps.hpp"
#include "alpaka/acc/AccFpgaSyclIntel.hpp"
#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/AccGpuCudaRt.hpp"
#include "alpaka/acc/AccGpuHipRt.hpp"
#include "alpaka/acc/AccGpuSyclIntel.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/acc/Traits.hpp"
// atomic
#include "alpaka/atomic/AtomicCpu.hpp"
#include "alpaka/atomic/AtomicGenericSycl.hpp"
#include "alpaka/atomic/AtomicNoOp.hpp"
#include "alpaka/atomic/AtomicOmpBuiltIn.hpp"
#include "alpaka/atomic/AtomicUniformCudaHipBuiltIn.hpp"
#include "alpaka/atomic/Op.hpp"
#include "alpaka/atomic/Traits.hpp"
// block
// shared
// dynamic
#include "alpaka/block/shared/dyn/BlockSharedMemDynGenericSycl.hpp"
#include "alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp"
#include "alpaka/block/shared/dyn/BlockSharedMemDynUniformCudaHipBuiltIn.hpp"
#include "alpaka/block/shared/dyn/Traits.hpp"
// static
#include "alpaka/block/shared/st/BlockSharedMemStGenericSycl.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStMember.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStMemberMasterSync.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStUniformCudaHipBuiltIn.hpp"
#include "alpaka/block/shared/st/Traits.hpp"
// sync
#include "alpaka/block/sync/BlockSyncBarrierOmp.hpp"
#include "alpaka/block/sync/BlockSyncBarrierThread.hpp"
#include "alpaka/block/sync/BlockSyncGenericSycl.hpp"
#include "alpaka/block/sync/BlockSyncNoOp.hpp"
#include "alpaka/block/sync/BlockSyncUniformCudaHipBuiltIn.hpp"
#include "alpaka/block/sync/Traits.hpp"
// core
#include "alpaka/core/Align.hpp"
#include "alpaka/core/AlignedAlloc.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/BarrierThread.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/ClipCast.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Debug.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/core/OmpSchedule.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/core/RemoveRestrict.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/core/ThreadPool.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/Unroll.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/core/Vectorize.hpp"
// dev
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/DevCpuSycl.hpp"
#include "alpaka/dev/DevCudaRt.hpp"
#include "alpaka/dev/DevFpgaSyclIntel.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/DevGpuSyclIntel.hpp"
#include "alpaka/dev/DevHipRt.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dev/cpu/Wait.hpp"
// dim
#include "alpaka/dim/DimArithmetic.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
// event
#include "alpaka/event/EventCpu.hpp"
#include "alpaka/event/EventCpuSycl.hpp"
#include "alpaka/event/EventCudaRt.hpp"
#include "alpaka/event/EventFpgaSyclIntel.hpp"
#include "alpaka/event/EventGenericSycl.hpp"
#include "alpaka/event/EventGpuSyclIntel.hpp"
#include "alpaka/event/EventHipRt.hpp"
#include "alpaka/event/Traits.hpp"
// extent
#include "alpaka/extent/Traits.hpp"
// idx
#include "alpaka/idx/Accessors.hpp"
#include "alpaka/idx/MapIdx.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/idx/bt/IdxBtGenericSycl.hpp"
#include "alpaka/idx/bt/IdxBtOmp.hpp"
#include "alpaka/idx/bt/IdxBtRefThreadIdMap.hpp"
#include "alpaka/idx/bt/IdxBtUniformCudaHipBuiltIn.hpp"
#include "alpaka/idx/bt/IdxBtZero.hpp"
#include "alpaka/idx/gb/IdxGbGenericSycl.hpp"
#include "alpaka/idx/gb/IdxGbRef.hpp"
#include "alpaka/idx/gb/IdxGbUniformCudaHipBuiltIn.hpp"
// kernel
#include "alpaka/kernel/TaskKernelCpuOmp2Blocks.hpp"
#include "alpaka/kernel/TaskKernelCpuOmp2Threads.hpp"
#include "alpaka/kernel/TaskKernelCpuSerial.hpp"
#include "alpaka/kernel/TaskKernelCpuSycl.hpp"
#include "alpaka/kernel/TaskKernelCpuTbbBlocks.hpp"
#include "alpaka/kernel/TaskKernelCpuThreads.hpp"
#include "alpaka/kernel/TaskKernelFpgaSyclIntel.hpp"
#include "alpaka/kernel/TaskKernelGenericSycl.hpp"
#include "alpaka/kernel/TaskKernelGpuCudaRt.hpp"
#include "alpaka/kernel/TaskKernelGpuHipRt.hpp"
#include "alpaka/kernel/TaskKernelGpuSyclIntel.hpp"
#include "alpaka/kernel/Traits.hpp"
// math
#include "alpaka/math/Complex.hpp"
#include "alpaka/math/MathGenericSycl.hpp"
#include "alpaka/math/MathStdLib.hpp"
#include "alpaka/math/MathUniformCudaHipBuiltIn.hpp"
// mem
#include "alpaka/mem/alloc/AllocCpuAligned.hpp"
#include "alpaka/mem/alloc/AllocCpuNew.hpp"
#include "alpaka/mem/alloc/Traits.hpp"
#include "alpaka/mem/buf/BufCpu.hpp"
#include "alpaka/mem/buf/BufCpuSycl.hpp"
#include "alpaka/mem/buf/BufCudaRt.hpp"
#include "alpaka/mem/buf/BufFpgaSyclIntel.hpp"
#include "alpaka/mem/buf/BufGenericSycl.hpp"
#include "alpaka/mem/buf/BufGpuSyclIntel.hpp"
#include "alpaka/mem/buf/BufHipRt.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/fence/MemFenceCpu.hpp"
#include "alpaka/mem/fence/MemFenceCpuSerial.hpp"
#include "alpaka/mem/fence/MemFenceGenericSycl.hpp"
#include "alpaka/mem/fence/MemFenceOmp2Blocks.hpp"
#include "alpaka/mem/fence/MemFenceOmp2Threads.hpp"
#include "alpaka/mem/fence/MemFenceUniformCudaHipBuiltIn.hpp"
#include "alpaka/mem/fence/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/mem/view/ViewConst.hpp"
#include "alpaka/mem/view/ViewPlainPtr.hpp"
#include "alpaka/mem/view/ViewStdArray.hpp"
#include "alpaka/mem/view/ViewStdVector.hpp"
#include "alpaka/mem/view/ViewSubView.hpp"
// meta
#include "alpaka/meta/Apply.hpp"
#include "alpaka/meta/CartesianProduct.hpp"
#include "alpaka/meta/Concatenate.hpp"
#include "alpaka/meta/CudaVectorArrayWrapper.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/meta/Filter.hpp"
#include "alpaka/meta/Fold.hpp"
#include "alpaka/meta/ForEachType.hpp"
#include "alpaka/meta/Functional.hpp"
#include "alpaka/meta/IntegerSequence.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/meta/IsArrayOrVector.hpp"
#include "alpaka/meta/IsStrictBase.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/meta/NonZero.hpp"
#include "alpaka/meta/Set.hpp"
#include "alpaka/meta/Transform.hpp"
#include "alpaka/meta/TypeListOps.hpp"
// offset
#include "alpaka/offset/Traits.hpp"
// platform
#include "alpaka/platform/PlatformCpu.hpp"
#include "alpaka/platform/PlatformCpuSycl.hpp"
#include "alpaka/platform/PlatformCudaRt.hpp"
#include "alpaka/platform/PlatformFpgaSyclIntel.hpp"
#include "alpaka/platform/PlatformGpuSyclIntel.hpp"
#include "alpaka/platform/PlatformHipRt.hpp"
#include "alpaka/platform/Traits.hpp"
// rand
#include "alpaka/rand/RandDefault.hpp"
#include "alpaka/rand/RandGenericSycl.hpp"
#include "alpaka/rand/RandPhilox.hpp"
#include "alpaka/rand/RandStdLib.hpp"
#include "alpaka/rand/RandUniformCudaHipRand.hpp"
#include "alpaka/rand/Traits.hpp"
// idx
#include "alpaka/idx/Traits.hpp"
// queue
#include "alpaka/queue/Properties.hpp"
#include "alpaka/queue/QueueCpuBlocking.hpp"
#include "alpaka/queue/QueueCpuNonBlocking.hpp"
#include "alpaka/queue/QueueCpuSyclBlocking.hpp"
#include "alpaka/queue/QueueCpuSyclNonBlocking.hpp"
#include "alpaka/queue/QueueCudaRtBlocking.hpp"
#include "alpaka/queue/QueueCudaRtNonBlocking.hpp"
#include "alpaka/queue/QueueFpgaSyclIntelBlocking.hpp"
#include "alpaka/queue/QueueFpgaSyclIntelNonBlocking.hpp"
#include "alpaka/queue/QueueGpuSyclIntelBlocking.hpp"
#include "alpaka/queue/QueueGpuSyclIntelNonBlocking.hpp"
#include "alpaka/queue/QueueHipRtBlocking.hpp"
#include "alpaka/queue/QueueHipRtNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"
// traits
#include "alpaka/traits/Traits.hpp"
// wait
#include "alpaka/wait/Traits.hpp"
// workdiv
#include "alpaka/workdiv/Traits.hpp"
#include "alpaka/workdiv/WorkDivHelpers.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"
// vec
#include "alpaka/vec/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
