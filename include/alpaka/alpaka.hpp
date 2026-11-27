/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Simd.hpp"
#include "alpaka/SimdWhereExpr.hpp"
#include "alpaka/UniqueId.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/api/cpu.hpp"
#include "alpaka/api/cuda/warp.hpp"
#include "alpaka/api/hip/warp.hpp"
#include "alpaka/api/host/warp.hpp"
#include "alpaka/api/oneApi.hpp"
#include "alpaka/api/oneApi/warp.hpp"
#include "alpaka/api/unifiedCudaHip.hpp"
#include "alpaka/apply.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/executor.hpp"
#include "alpaka/fn.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/intrinsic.hpp"
#include "alpaka/math.hpp"
#include "alpaka/math/Complex.hpp"
#include "alpaka/math/constants.hpp"
#include "alpaka/mem/BoundaryIter.hpp"
#include "alpaka/mem/LinearizedIdxGenerator.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/WorkerGroup.hpp"
#include "alpaka/onAcc/atomic.hpp"
#include "alpaka/onAcc/globalMem.hpp"
#include "alpaka/onAcc/interface.hpp"
#include "alpaka/onAcc/internal/interfaceImpl.hpp"
#include "alpaka/onAcc/memFence.hpp"
#include "alpaka/onAcc/range.hpp"
#include "alpaka/onAcc/tag.hpp"
#include "alpaka/onAcc/warp.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"
#include "alpaka/onHost/EventPolicyList.hpp"
#include "alpaka/onHost/MemoryPolicyList.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/QueuePolicyList.hpp"
#include "alpaka/onHost/algo/concurrent.hpp"
#include "alpaka/onHost/algo/iota.hpp"
#include "alpaka/onHost/algo/reduce.hpp"
#include "alpaka/onHost/algo/scan.hpp"
#include "alpaka/onHost/algo/transform.hpp"
#include "alpaka/onHost/algo/transformReduce.hpp"
#include "alpaka/onHost/demangledName.hpp"
#include "alpaka/onHost/executeForEach.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/onHost/mem/stdContainer.hpp"
#include "alpaka/rand/distribution/NormalReal.hpp"
#include "alpaka/rand/distribution/UniformReal.hpp"
#include "alpaka/rand/engine/philox/philox.hpp"
#include "alpaka/simd/math.hpp"
#include "alpaka/simd/simdized.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

/** main alpaka namespace.
 *
 * namespace onHost::* contains all functionality which is usable on the host CPU controller thread.
 * namespace onAcc::* contains all functionality which is usable on the accelerator compute device from within a
 * kernel. namespace alpaka contains all functionality which is generic and can be used from within the host controller
 * thread and within compute device kernels.
 */
namespace alpaka
{
}
