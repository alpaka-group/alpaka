/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#pragma once

namespace alpaka
{
    //! Alias for the default accelerator used by examples. From a list of
    //! all accelerators the first one which is enabled is chosen.
    //! AccCpuSerial is selected last.
    template<class TDim, class TIdx>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using ExampleDefaultAcc = alpaka::AccGpuCudaRt<TDim, TIdx>;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using ExampleDefaultAcc = alpaka::AccGpuHipRt<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    using ExampleDefaultAcc = alpaka::AccCpuOmp2Blocks<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    using ExampleDefaultAcc = alpaka::AccCpuTbbBlocks<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    using ExampleDefaultAcc = alpaka::AccCpuOmp2Threads<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    using ExampleDefaultAcc = alpaka::AccCpuThreads<TDim, TIdx>;
#elif defined(ALPAKA_ACC_ANY_BT_OMP5_ENABLED)
    using ExampleDefaultAcc = alpaka::AccOmp5<TDim, TIdx>;
#elif defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED)
    using ExampleDefaultAcc = alpaka::AccOacc<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    using ExampleDefaultAcc = alpaka::AccCpuSerial<TDim, TIdx>;
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI)
#    if defined(ALPAKA_SYCL_ONEAPI_CPU)
    using ExampleDefaultAcc = alpaka::experimental::AccCpuSyclIntel<TDim, TIdx>;
#    elif defined(ALPAKA_SYCL_ONEAPI_FPGA)
    using ExampleDefaultAcc = alpaka::experimental::AccFpgaSyclIntel<TDim, TIdx>;
#    elif defined(ALPAKA_SYCL_ONEAPI_GPU)
    using ExampleDefaultAcc = alpaka::experimental::AccGpuSyclIntel<TDim, TIdx>;
#    endif
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)
    using ExampleDefaultAcc = alpaka::experimental::AccFpgaSyclXilinx<TDim, TIdx>;
#else
    class ExampleDefaultAcc;
#    warning "No supported backend selected."
#endif
} // namespace alpaka
