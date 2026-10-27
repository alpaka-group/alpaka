/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * This test validate if a M-dimension kernel can be called with a FrameSpec and ThreadSpec.
 * Additionally the M-dimensional memcpy and memset is tested too.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

struct LastSetDataBlockIdx
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto out,
        alpaka::concepts::Vector auto extentMd) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(extentMd)))
        {
            out[i] = i;
        }
    }
};

template<alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_FrameSize>
struct Case
{
    Case(T_Extents extents, T_FrameSize frameSize) : extents(extents), frameSize(frameSize)
    {
    }

    T_Extents extents;
    T_FrameSize frameSize;
};

void validate(auto& queue, auto& device, auto exec, auto testCase)
{
    Vec extentMd = testCase.extents;
    INFO(
        "exec=" << onHost::demangledName(exec) << " extents=" << testCase.extents
                << " frame size=" << testCase.frameSize);
    auto dBuff = onHost::alloc<Vec<uint32_t, extentMd.dim()>>(device, extentMd);

    auto hBuff = onHost::allocHostLike(dBuff);

    // fill with non-zero values
    onHost::fill(queue, dBuff, extentMd);

    onHost::wait(queue);
    auto frameSize = testCase.frameSize;
    queue.enqueue(
        onHost::FrameSpec{divExZero(extentMd, frameSize), frameSize, exec},
        KernelBundle{LastSetDataBlockIdx{}, dBuff, extentMd});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    // validate that each data entry has its corresponding MD-element index
    meta::ndLoopIncIdx(extentMd, [&](auto idx) { CHECK(hBuff[idx] == idx); });

    onHost::memset(queue, dBuff, 0u);
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    // validate that all data is zero
    meta::ndLoopIncIdx(extentMd, [&](auto idx) { CHECK(hBuff[idx] == ALPAKA_TYPEOF(extentMd)::fill(0)); });
}

TEMPLATE_LIST_TEST_CASE("kernelCallMD", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    auto testCfg = std::make_tuple(
        Case{Vec{3u}, Vec{2u}},
        Case{Vec{4u, 7u}, Vec{2u, 4u}},
        Case{Vec{4u, 8u, 13}, Vec{2u, 4u, 8u}},
        Case{Vec{4u, 8u, 16, 31}, Vec{2u, 4u, 8u, 4u}},
        Case{Vec{4u, 8u, 16, 8u, 3u}, Vec{2u, 4u, 8u, 4u, 2u}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u}, Vec{2u, 4u, 8u, 4u, 2u, 3u}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u, 2u}, Vec{2u, 4u, 8u, 4u, 2u, 3u, 2u}},
        Case{Vec{3u}, CVec<uint32_t, 2u>{}},
        Case{Vec{4u, 7u}, CVec<uint32_t, 2u, 4u>{}},
        Case{Vec{4u, 8u, 13}, CVec<uint32_t, 2u, 4u, 8u>{}},
        Case{Vec{4u, 8u, 16, 31}, CVec<uint32_t, 2u, 4u, 8u, 4u>{}},
        Case{Vec{4u, 8u, 16, 8u, 3u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u>{}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u, 3u>{}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u, 2u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u, 3u, 2u>{}});

    std::apply([&](auto... testCase) { (validate(queue, device, exec, testCase), ...); }, testCfg);
}
