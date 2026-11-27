/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/api/hip/Api.hpp"
#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/oneApi/Api.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/trait.hpp"

#include <algorithm>
#include <type_traits>

namespace alpaka
{
    /** provides the API used during the execution of the current code path
     *
     * @attention if api::host os returned it can also mean that this method was called within the host controlling
     * workflow and not within a kernel running on a CPU device.
     */
    constexpr auto thisApi()
    {
#if ALPAKA_LANG_SYCL && ALPAKA_LANG_ONEAPI && defined(__SYCL_DEVICE_ONLY__)
        return api::oneApi;
#elif ALPAKA_LANG_CUDA && (ALPAKA_COMP_CLANG_CUDA || ALPAKA_COMP_NVCC) && __CUDA_ARCH__
        return api::cuda;
#elif ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
        return api::hip;
#else
        return api::host;
#endif
    }

    /** True only while compiling device code for an NVIDIA or AMD GPU.
     *
     * Covers native CUDA/HIP and oneAPI/SYCL kernels lowered to those devices. The host pass of those builds and
     * every other target return false.
     */
    constexpr bool isNvidiaOrAmdDeviceCode()
    {
#if ALPAKA_ARCH_PTX || ALPAKA_ARCH_AMD
        return true;
#elif defined(__SYCL_DEVICE_ONLY__) && (defined(__NVPTX__) || defined(__AMDGCN__))
        return true;
#else
        return false;
#endif
    }

    namespace onHost
    {
        constexpr auto apis = std::make_tuple(api::host, api::cuda, api::hip, api::oneApi);

        constexpr auto enabledApis = meta::filter([](auto api) constexpr { return isPlatformAvailable(api); }, apis);
    } // namespace onHost

    namespace api
    {
        constexpr bool operator==(alpaka::concepts::Api auto lhs, alpaka::concepts::Api auto rhs)
        {
            return std::is_same_v<ALPAKA_TYPEOF(lhs), ALPAKA_TYPEOF(rhs)>;
        }

        constexpr bool operator!=(alpaka::concepts::Api auto lhs, alpaka::concepts::Api auto rhs)
        {
            return !(lhs == rhs);
        }
    } // namespace api
} // namespace alpaka
