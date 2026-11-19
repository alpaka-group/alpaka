/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

template<typename T_Data>
int validateResult(
    auto const& queue,
    concepts::IDataSource<T_Data> auto const& inputData,
    concepts::IDataSource<T_Data> auto const& bufY,
    std::size_t numElements,
    onHost::internal::ScanType scanType)
{
    // Copy back the result
    auto bufHostY = onHost::allocHost<T_Data>(numElements);

    onHost::memcpy(queue, bufHostY, bufY);
    onHost::wait(queue);

    // validate results
    int falseResults = 0;
    static constexpr int maxPrintFalseResults = 20;

    auto groundtruth = onHost::allocHost<T_Data>(numElements);

    switch(scanType)
    {
    case onHost::internal::exclusiveScan:
        std::exclusive_scan(inputData.data(), inputData.data() + numElements, groundtruth.data(), 0);
        break;
    case onHost::internal::inclusiveScan:
        std::inclusive_scan(inputData.data(), inputData.data() + numElements, groundtruth.data());
        break;
    }

    for(std::size_t i = 0u; i < numElements; ++i)
    {
        T_Data const& computedY = bufHostY[i];
        T_Data const& correctResult = groundtruth[i];

        if(computedY != correctResult)
        {
            if(falseResults < maxPrintFalseResults)
                UNSCOPED_INFO("bufY[" << i << "] == " << computedY << " != " << correctResult);
            ++falseResults;
        }
    }

    if(falseResults == 0)
    {
        return EXIT_SUCCESS;
    }

    UNSCOPED_INFO(
        "Found " << falseResults << " false results, printed no more than " << maxPrintFalseResults
                 << ". Execution results incorrect!");
    return EXIT_FAILURE;
}

template<typename T_Data>
void testExclusiveScan(
    concepts::Executor auto& exec,
    alpaka::onHost::concepts::Device auto& computeDev,
    auto& computeQueue,
    concepts::IView<T_Data> auto& hostIn,
    concepts::IView<T_Data> auto& inBuf,
    concepts::IView<T_Data> auto& outBuf)
{
    auto numEl = hostIn.getExtents().x();

    // exclusive scan, no buffer
    INFO("exclusive scan, no buffer");
    onHost::memcpy(computeQueue, inBuf, hostIn);
    onHost::exclusiveScan(computeQueue, exec, outBuf, inBuf);
    auto res = validateResult<T_Data>(computeQueue, hostIn, outBuf, numEl, onHost::internal::exclusiveScan);
    CHECK(res == EXIT_SUCCESS);

    // exclusive scan, in-place
    INFO("exclusive scan, no buffer, in-place");
    onHost::exclusiveScanInPlace(computeQueue, exec, inBuf);
    res = validateResult<T_Data>(computeQueue, hostIn, inBuf, numEl, onHost::internal::exclusiveScan);
    CHECK(res == EXIT_SUCCESS);


    onHost::SharedBuffer intermediateBuffer
        = onHost::alloc<std::byte>(computeDev, onHost::getScanBufferSize<T_Data>(inBuf.getExtents()));

    // exclusive scan, with buffer
    INFO("exclusive scan, with buffer");
    onHost::memcpy(computeQueue, inBuf, hostIn);
    onHost::exclusiveScan(computeQueue, exec, intermediateBuffer, outBuf, inBuf);
    res = validateResult<T_Data>(computeQueue, hostIn, outBuf, numEl, onHost::internal::exclusiveScan);
    CHECK(res == EXIT_SUCCESS);

    // exclusive scan, in-place with buffer
    INFO("exclusive scan, with buffer, in-place");
    onHost::exclusiveScanInPlace(computeQueue, exec, intermediateBuffer, inBuf);
    res = validateResult<T_Data>(computeQueue, hostIn, inBuf, numEl, onHost::internal::exclusiveScan);
    CHECK(res == EXIT_SUCCESS);
}

template<typename T_Data>
void testInclusiveScan(
    concepts::Executor auto& exec,
    alpaka::onHost::concepts::Device auto& computeDev,
    auto& computeQueue,
    concepts::IView<T_Data> auto& hostIn,
    concepts::IView<T_Data> auto& inBuf,
    concepts::IView<T_Data> auto& outBuf)
{
    auto numEl = hostIn.getExtents().x();

    // inclusive scan, no buffer
    INFO("inclusive scan, no buffer");
    onHost::memcpy(computeQueue, inBuf, hostIn);
    onHost::inclusiveScan(computeQueue, exec, outBuf, inBuf);
    auto res = validateResult<T_Data>(computeQueue, hostIn, outBuf, numEl, onHost::internal::inclusiveScan);
    CHECK(res == EXIT_SUCCESS);

    // inclusive scan, in-place
    INFO("inclusive scan, no buffer, in-place");
    onHost::inclusiveScanInPlace(computeQueue, exec, inBuf);
    res = validateResult<T_Data>(computeQueue, hostIn, inBuf, numEl, onHost::internal::inclusiveScan);
    CHECK(res == EXIT_SUCCESS);


    onHost::SharedBuffer intermediateBuffer
        = onHost::alloc<std::byte>(computeDev, onHost::getScanBufferSize<T_Data>(inBuf.getExtents()));

    // inclusive scan, with buffer
    INFO("inclusive scan, with buffer");
    onHost::memcpy(computeQueue, inBuf, hostIn);
    onHost::inclusiveScan(computeQueue, exec, intermediateBuffer, outBuf, inBuf);
    res = validateResult<T_Data>(computeQueue, hostIn, outBuf, numEl, onHost::internal::inclusiveScan);
    CHECK(res == EXIT_SUCCESS);

    // inclusive scan, in-place with buffer
    INFO("inclusive scan, with buffer, in-place");
    onHost::inclusiveScanInPlace(computeQueue, exec, intermediateBuffer, inBuf);
    res = validateResult<T_Data>(computeQueue, hostIn, inBuf, numEl, onHost::internal::inclusiveScan);
    CHECK(res == EXIT_SUCCESS);
}

template<typename T_Data>
void prepareTest(auto cfg, concepts::Vector auto extents)
{
    using DataType = T_Data;
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue computeQueue = computeDev.makeQueue();

    // allocate buffers
    onHost::SharedBuffer inputBuf = onHost::alloc<T_Data>(computeDev, extents);
    onHost::SharedBuffer outputBuf = onHost::alloc<T_Data>(computeDev, extents);

    // allocate host buffers
    onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(inputBuf);
    onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(outputBuf);

    // initialize with the linearized index
    DataType iotaCounter = 0;
    for(auto& value : hostBufferIota)
    {
        value = iotaCounter;
        ++iotaCounter;
    }

    testExclusiveScan<T_Data>(exec, computeDev, computeQueue, hostBufferIota, inputBuf, outputBuf);
    testInclusiveScan<T_Data>(exec, computeDev, computeQueue, hostBufferIota, inputBuf, outputBuf);
}

TEMPLATE_LIST_TEST_CASE("scan", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // different extents for testing
    auto extentList = std::make_tuple(Vec{12}, Vec{1024}, Vec{654321}, Vec{20'971'520});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents), ...); }, extentList);
}
