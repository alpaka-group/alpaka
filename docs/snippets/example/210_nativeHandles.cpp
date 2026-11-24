/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#if ALPAKA_LANG_CUDA
#    include <cuda_runtime.h>
#endif

#if ALPAKA_LANG_HIP
#    include <hip/hip_runtime.h>
#endif

#if ALPAKA_LANG_ONEAPI
#    include <sycl/sycl.hpp>
#endif

using namespace alpaka;

#if ALPAKA_LANG_CUDA
TEST_CASE("native handle link - cuda", "[docs]")
{
    // BEGIN-TUTORIAL-nativeCuda
    auto sel = onHost::makeDeviceSelector(api::cuda, deviceKind::nvidiaGpu);
    if(!sel.isAvailable())
        return;

    uint32_t const nativeDevice = 0;
    REQUIRE(cudaSetDevice(static_cast<int>(nativeDevice)) == cudaSuccess);

    cudaStream_t nativeQueue{};
    REQUIRE(cudaStreamCreateWithFlags(&nativeQueue, cudaStreamNonBlocking) == cudaSuccess);

    {
        onHost::Device dev = sel.linkDevice(nativeDevice, true);
        onHost::Queue q = dev.linkQueue(queueKind::nonBlocking, nativeQueue, false);

        Vec const extents = Vec{8u};
        auto buffer = onHost::alloc<int>(dev, extents);
        onHost::iota(q, 0, buffer);
        auto host = onHost::allocHost<int>(extents);
        onHost::memcpy(q, host, buffer);
        onHost::wait(q);
        CHECK(host[0] == 0);

        dev.unlinkQueue(nativeQueue);
        sel.unlinkDevice(nativeDevice);
    }

    // The alpaka handles are destroyed above; only now the native objects may be destroyed.
    REQUIRE(cudaStreamDestroy(nativeQueue) == cudaSuccess);
    // END-TUTORIAL-nativeCuda
}
#endif

#if ALPAKA_LANG_HIP
TEST_CASE("native handle link - hip", "[docs]")
{
    // BEGIN-TUTORIAL-nativeHip
    auto sel = onHost::makeDeviceSelector(api::hip, deviceKind::amdGpu);
    if(!sel.isAvailable())
        return;

    uint32_t const nativeDevice = 0;
    REQUIRE(hipSetDevice(static_cast<int>(nativeDevice)) == hipSuccess);

    hipStream_t nativeQueue{};
    REQUIRE(hipStreamCreateWithFlags(&nativeQueue, hipStreamNonBlocking) == hipSuccess);

    {
        onHost::Device dev = sel.linkDevice(nativeDevice, true);
        onHost::Queue q = dev.linkQueue(queueKind::nonBlocking, nativeQueue, false);

        Vec const extents = Vec{8u};
        auto buffer = onHost::alloc<int>(dev, extents);
        onHost::iota(q, 0, buffer);
        auto host = onHost::allocHost<int>(extents);
        onHost::memcpy(q, host, buffer);
        onHost::wait(q);
        CHECK(host[0] == 0);

        dev.unlinkQueue(nativeQueue);
        sel.unlinkDevice(nativeDevice);
    }

    // The alpaka handles are destroyed above; only now the native objects may be destroyed.
    REQUIRE(hipStreamDestroy(nativeQueue) == hipSuccess);
    // END-TUTORIAL-nativeHip
}
#endif

#if ALPAKA_LANG_ONEAPI
TEST_CASE("native handle link - oneapi", "[docs]")
{
    // BEGIN-TUTORIAL-nativeOneApi
    auto sel = onHost::makeDeviceSelector(api::oneApi, deviceKind::intelGpu);
    if(!sel.isAvailable())
        return;

    sycl::device const syclDevice{sycl::gpu_selector_v};
    sycl::context const syclContext{syclDevice};
    std::pair<sycl::device, sycl::context> const nativeDevice{syclDevice, syclContext};
    sycl::queue nativeQueue{syclContext, syclDevice, sycl::property::queue::in_order{}};

    {
        onHost::Device dev = sel.linkDevice(nativeDevice, true);
        onHost::Queue q = dev.linkQueue(queueKind::nonBlocking, nativeQueue, false);

        Vec const extents = Vec{8u};
        auto buffer = onHost::alloc<int>(dev, extents);
        onHost::iota(q, 0, buffer);
        auto host = onHost::allocHost<int>(extents);
        onHost::memcpy(q, host, buffer);
        onHost::wait(q);
        CHECK(host[0] == 0);

        dev.unlinkQueue(nativeQueue);
        sel.unlinkDevice(nativeDevice);
    }
    // END-TUTORIAL-nativeOneApi
}
#endif
