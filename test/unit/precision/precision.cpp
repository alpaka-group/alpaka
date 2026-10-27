/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/meta/NdLoop.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

/** @file Test if we can execute a kernel with different precisions for kernel start parameters, in kernel index loop
 * and memory index types.
 *
 * Since vectors used in alpaka not perform implicit type conversion, except for operators where left or right hand
 * side is a scalar and if we do not lose precision or signedness, the tests guard against mistakes within alpaka due
 * to `auto` usage.
 */

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, exec::enabledExecutors))>;

struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        using MemScalarType = typename alpaka::GetValueType_t<ALPAKA_TYPEOF(out)>::value_type;

        // rerun the tests with automatic derived loop index type
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = pCast<MemScalarType>(i);
        }
    }
};

template<typename T_MemIdxType>
void iotaTest(auto& queue, auto exec, auto kernel, auto const extents, auto frameSize)
{
    auto dBuff = onHost::alloc<Vec<T_MemIdxType, ALPAKA_TYPEOF(extents)::dim()>>(queue.getDevice(), extents);
    auto hBuff = onHost::allocHostLike(dBuff);

    using KenelIdxScalarType = typename ALPAKA_TYPEOF(frameSize)::value_type;
    onHost::wait(queue);
    queue.enqueue(
        FrameSpec{pCast<KenelIdxScalarType>(extents) / frameSize, frameSize, exec},
        KernelBundle{kernel, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    alpaka::concepts::IMdSpan auto mdSpan = hBuff.getMdSpan();

    meta::ndLoopIncIdx(extents, [&](auto idx) { CHECK(idx == pCast<T_MemIdxType>(mdSpan[idx])); });
}

enum Sign : bool
{
    signedIdx = true,
    unsignedIdx = false
};

template<Sign T_signedMemIdx, Sign T_signedKernelIdx>
void callTests(auto cfg)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device device = test::getDevice(deviceExec);
    alpaka::concepts::Executor auto exec = test::getExecutor(deviceExec);
    Queue queue = device.makeQueue();

    IotaKernelND iotaKernelND;
    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }
    // half of the combinations
    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType>(queue, exec, iotaKernelND, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType>(
            queue,
            exec,
            iotaKernelND,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }
}

TEMPLATE_LIST_TEST_CASE("kernel_precision_iotaNd", "", TestApis)
{
    auto cfg = TestType::makeDict();
    // test signed-ness
    callTests<signedIdx, signedIdx>(cfg);
    callTests<unsignedIdx, signedIdx>(cfg);
    callTests<signedIdx, unsignedIdx>(cfg);
    callTests<unsignedIdx, unsignedIdx>(cfg);
}

template<typename T_LoopIdxType>
struct IotaKernelNDForceCast
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        using MemScalarType = typename alpaka::GetValueType_t<ALPAKA_TYPEOF(out)>::value_type;
        for(auto i : onAcc::makeIdxMap<T_LoopIdxType>(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = pCast<MemScalarType>(i);
        }
    }
};

template<typename T_MemIdx, typename T_LoopIdx>
void callForceCastTests(auto cfg)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device device = test::getDevice(deviceExec);
    alpaka::concepts::Executor auto exec = test::getExecutor(deviceExec);
    Queue queue = device.makeQueue();

    IotaKernelNDForceCast<T_LoopIdx> iotaKernelNDForceCast;

    iotaTest<T_MemIdx>(queue, exec, iotaKernelNDForceCast, Vec<T_MemIdx, 1u>{15u}, Vec<uint32_t, 1u>{7u});
    iotaTest<T_MemIdx>(queue, exec, iotaKernelNDForceCast, Vec<T_MemIdx, 2u>{8u, 16u}, Vec<uint32_t, 2u>{2u, 4u});
    iotaTest<T_MemIdx>(
        queue,
        exec,
        iotaKernelNDForceCast,
        Vec<T_MemIdx, 3u>{3u, 8u, 16u},
        Vec<uint32_t, 3u>{2u, 2u, 4u});
}

TEMPLATE_LIST_TEST_CASE("kernel_precision_iotaNd_force_cast", "", TestApis)
{
    auto cfg = TestType::makeDict();

    // The first type is the index_type of the extents of the MdSpan.
    // The second type is used to convert the index type of the idx object to this type. This is done using the command
    // `makeIdxMap<T_LoopIdxType>()`.
    // Therefore, the second type must be castable to the first type without precision lost in order to satisfy the
    // IndexVec concept of the MdSpan access operator.
    callForceCastTests<uint32_t, uint32_t>(cfg);
    callForceCastTests<uint64_t, uint32_t>(cfg);
    callForceCastTests<uint64_t, uint32_t>(cfg);
    callForceCastTests<int64_t, int32_t>(cfg);
    callForceCastTests<int64_t, uint32_t>(cfg);

    // does not work, because the int32_t could be negative
    // callForceCastTests<uint32_t, int32_t>(cfg);
    // does not work, because the value of uint32_t could be bigger than 2^31-1
    // callForceCastTests<int32_t, uint32_t>(cfg);
    // does not work, because uint64_t is bigger than uint32_t
    // callForceCastTests<uint32_t, uint64_t>(cfg);
    // does not work, because the int32_t could be negative
    // callForceCastTests<uint64_t, int32_t>(cfg);
}
