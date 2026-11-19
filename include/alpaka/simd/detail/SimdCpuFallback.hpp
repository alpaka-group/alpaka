/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/acc/AccCpuSerial.hpp>

#include <cstddef>
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
#    include <alpaka/acc/AccCpuThreads.hpp>
#endif
#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
#    include <alpaka/acc/AccCpuOmp2Threads.hpp>
#endif
#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#    include <alpaka/acc/AccCpuTbbBlocks.hpp>
#endif

#include "alpaka/simd/detail/SimdCpu.hpp"

namespace alpaka::simd
{
    // NOTE: Fallback core implementation is solely in SimdCpu.hpp.
    // Fallback is actually an array based auto-vectorization. First simd-policy level is using
    // std::experimental::simd, second is fallback. This header only provides accelerator inheritance specializations
    // to avoid duplication.

    // Accelerator specializations using inheritance for code reuse

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuThreads<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;

        ALPAKA_FN_HOST_ACC PortableSimd(Base const& other) : Base(other)
        {
        }

        constexpr std::size_t size() const noexcept
        {
            return Base::size();
        }
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuOmp2Blocks<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;

        ALPAKA_FN_HOST_ACC PortableSimd(Base const& other) : Base(other)
        {
        }

        constexpr std::size_t size() const noexcept
        {
            return Base::size();
        }
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuOmp2Threads<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;

        ALPAKA_FN_HOST_ACC PortableSimd(Base const& other) : Base(other)
        {
        }

        constexpr std::size_t size() const noexcept
        {
            return Base::size();
        }
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuTbbBlocks<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;

        ALPAKA_FN_HOST_ACC PortableSimd(Base const& other) : Base(other)
        {
        }

        constexpr std::size_t size() const noexcept
        {
            return Base::size();
        }
    };
#endif

} // namespace alpaka::simd
