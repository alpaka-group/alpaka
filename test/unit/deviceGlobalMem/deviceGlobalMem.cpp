/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/* We can not check passing the attributes 'const', 'static' or 'constexpr' because this is not supported by OneApi
 * 2025.2 with AMD gpus.
 * Therefore, we are testing only empty attributes and 'inline' attribute to fucus on supported code only.
 */
ALPAKA_DEVICE_GLOBAL_EXTERN(, (alpaka::Vec<uint32_t, 2u>), initialisedVector);
ALPAKA_DEVICE_GLOBAL(, (alpaka::Vec<uint32_t, 2u>), initialisedVector, 42u, 43u);

ALPAKA_DEVICE_GLOBAL(, uint32_t, initialisedScalar, 43u);
ALPAKA_DEVICE_GLOBAL(, uint32_t[2], fixedSizedArray, {44u, 45u});
ALPAKA_DEVICE_GLOBAL(, uint32_t[2][3], fixedSizedArray2D, {{9u, 5u}, {6u, 11u, 45u}});

struct DeviceGlobalMemKernelVec
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = initialisedVector.get().y();
        }
    }
};

struct DeviceGlobalMemKernelScalar
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = initialisedScalar.get();
        }
    }
};

struct DeviceGlobalMemKernelCArray
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = fixedSizedArray.get()[0];
        }
    }
};

struct DeviceGlobalMemKernelCArray2D
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = fixedSizedArray2D.get()[Vec<size_t, 2>{1, 2}];
        }
    }
};

TEMPLATE_LIST_TEST_CASE("device global mem", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    alpaka::concepts::Executor auto exec = test::getExecutor(deviceExec);

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{4u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);
    {
        queue.enqueue(
            FrameSpec{numBlocks, blockExtent, exec},
            KernelBundle{DeviceGlobalMemKernelVec{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(42 == ptr[i]);
        }
    }

    // scalar
    {
        queue.enqueue(
            FrameSpec{numBlocks, blockExtent, exec},
            KernelBundle{DeviceGlobalMemKernelScalar{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(43 == ptr[i]);
        }
    }

    // C array
    {
        queue.enqueue(
            FrameSpec{numBlocks, blockExtent, exec},
            KernelBundle{DeviceGlobalMemKernelCArray{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(44 == ptr[i]);
        }
    }

    // C array 2D
    {
        queue.enqueue(
            FrameSpec{numBlocks, blockExtent, exec},
            KernelBundle{DeviceGlobalMemKernelCArray2D{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(45 == ptr[i]);
        }
    }
}

// The initial value will be overwritten
ALPAKA_DEVICE_GLOBAL(, uint32_t, globalScalar, 42u);

struct DeviceGlobalMemCpyScalarKernel
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[idx] = globalScalar.get();
        }
    }
};

ALPAKA_DEVICE_GLOBAL(, uint32_t[2u][3u], global2DCArray, {{9u, 5u}, {6u, 11u, 45u}});

struct DeviceGlobalMemCpyCArray2DKernel
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[idx] = global2DCArray.get()[Vec{1u, 2u}];
        }
    }
};

TEMPLATE_LIST_TEST_CASE("device global mem copy", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    alpaka::concepts::Executor auto exec = test::getExecutor(deviceExec);

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{4u};
    constexpr Vec dataExtent = 13u;
    INFO("device name: " << device.getName());
    INFO("executor   : " << exec.getName());

    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);
    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);
    {
        /* Get the value from the host representation and increase it because if we run with multiple host executors we
         * need to be sure we test unique values for each executor. */
        uint32_t newValue = globalScalar.get() + 1u;
        if(queue.getApi() != api::host)
            globalScalar.get() = newValue - 2u;

        uint32_t val = newValue;
        // check that we can copy from host to device
        onHost::memcpy(queue, globalScalar, &val);
        // reset host value
        queue.enqueueHostFn([&]() { val = 0u; });
        // check that we can copy from device to host
        onHost::memcpy(queue, &val, globalScalar);
        queue.enqueue(FrameSpec{numBlocks, blockExtent, exec}, KernelBundle{DeviceGlobalMemCpyScalarKernel{}, dBuff});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);
        // check that copy from device to host worked
        CHECK(val == newValue);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(newValue == hBuff[i]);
            if(queue.getApi() != api::host)
            {
                // the host and device representation must differ because they are manipulated independently
                CHECK(globalScalar.get() != hBuff[i]);
            }
        }
    }
    {
        /* Get the value from the host representation and increase it because if we run with multiple host executors we
         * need to be sure we test unique values for each executor. */
        uint32_t newValue = global2DCArray.get()[Vec{1u, 2u}] + 1u;
        if(queue.getApi() != api::host)
            global2DCArray.get()[Vec{1u, 2u}] = newValue - 2u;

        uint32_t val[2u][3u] = {{9u, 5u}, {6u, 11u, 45u}};
        val[1][2] = newValue;
        onHost::memcpy(queue, global2DCArray, val);
        // reset host value
        queue.enqueueHostFn([&]() { val[1][2] = 0u; });
        onHost::memcpy(queue, val, global2DCArray);
        queue.enqueue(
            FrameSpec{numBlocks, blockExtent, exec},
            KernelBundle{DeviceGlobalMemCpyCArray2DKernel{}, dBuff});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);
        // check that copy from device to host worked
        CHECK(val[1][2] == newValue);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(newValue == hBuff[i]);
            if(queue.getApi() != api::host)
            {
                // the host and device representation must differ because they are manipulated independently
                CHECK(global2DCArray.get()[Vec{1u, 2u}] != hBuff[i]);
            }
        }
    }
}
