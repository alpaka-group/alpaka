/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */


#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <functional>

using namespace alpaka;

// BEGIN-TUTORIAL-transformFunctor
struct SquareValueScalar
{
    // int is the data type of the input buffer and float the type of the output buffer
    ALPAKA_FN_ACC auto operator()(int const& value) const -> float
    {
        return static_cast<float>(value * value);
    }
};

// END-TUTORIAL-transformFunctor

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - transform", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 10;
    Vec extents(size);

    // BEGIN-TUTORIAL-transformCall
    // we are free to combine different data types for the in- and output
    auto inputBuffer = onHost::alloc<int>(device, extents);
    auto outputBuffer = onHost::alloc<float>(device, extents);

    onHost::fill(queue, inputBuffer, 2);
    onHost::fill(queue, outputBuffer, 0.f);

    // call transform with ScalarFunc wrapper
    onHost::transform(queue, exec, outputBuffer, ScalarFunc{SquareValueScalar{}}, inputBuffer);
    // END-TUTORIAL-transformCall


    auto hostOutput = onHost::allocHost<float>(extents);
    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < size; ++i)
    {
        REQUIRE(hostOutput[i] == 4.f);
    }
}

// BEGIN-TUTORIAL-transformFunctorDefaultFunc
struct SquareValue
{
    ALPAKA_FN_ACC auto operator()(concepts::Simd auto const& value) const -> concepts::Simd auto
    {
        // SIMD operation
        // For example, int is 32 bit and the processor support AVX (128 bit),
        // 4 multiplications are done in one operation.
        return value * value;
    }
};

// END-TUTORIAL-transformFunctorDefaultFunc

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - transform default functor", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 10;
    Vec extents(size);

    // BEGIN-TUTORIAL-transformCallDefaultFunc
    auto inputBuffer = onHost::alloc<int>(device, extents);
    auto outputBuffer = onHost::alloc<int>(device, extents);

    onHost::fill(queue, inputBuffer, 2);
    onHost::fill(queue, outputBuffer, 0);

    // call transform without wrapper -> enables SIMD support
    onHost::transform(queue, exec, outputBuffer, SquareValue{}, inputBuffer);
    // END-TUTORIAL-transformCallDefaultFunc

    auto hostOutput = onHost::allocHost<int>(extents);
    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < size; ++i)
    {
        REQUIRE(hostOutput[i] == 4);
    }
}

// BEGIN-TUTORIAL-transformFunctorStencilFunc
struct SumNeighbors
{
    ALPAKA_FN_ACC auto operator()(concepts::SimdPtr auto const& in) const
    {
        using ResultType = ALPAKA_TYPEOF(in.load());
        ResultType result = ResultType::fill(0);

        using VecIdxType = typename ALPAKA_TYPEOF(in)::IdxType;
        // the functor is written for a 2D stencil
        static_assert(VecIdxType::dim() == 2);

        // top neighbor
        result += in[VecIdxType{-1, 0}].load();
        // right neighbor
        result += in[VecIdxType{0, 1}].load();
        // bottom neighbor
        result += in[VecIdxType{1, 0}].load();
        // left neighbor
        result += in[VecIdxType{0, -1}].load();

        return result;
    }
};

// END-TUTORIAL-transformFunctorStencilFunc

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - transform stencil functor", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr int size = 10;

    // BEGIN-TUTORIAL-transformCallStencilFunc
    Vec extents(size, size);
    auto inputBuffer = onHost::alloc<int>(device, extents);
    auto outputBuffer = onHost::alloc<int>(device, extents);

    // we use a 2D stencil code
    STATIC_REQUIRE(inputBuffer.dim() == 2);
    STATIC_REQUIRE(outputBuffer.dim() == 2);

    onHost::fill(queue, inputBuffer, 1);
    onHost::fill(queue, outputBuffer, 0);

    onHost::transform(
        queue,
        exec,
        // the user must provide a subview that takes the halo into account.
        outputBuffer.getView().getSubView(Vec{1, 1}, Vec{size - 2, size - 2}),
        // use stencil wrapper
        StencilFunc{SumNeighbors{}},
        // both input and output needs to take the halo into account
        inputBuffer.getView().getSubView(Vec{1, 1}, Vec{size - 2, size - 2}));
    // END-TUTORIAL-transformCallStencilFunc

    auto hostOutput = onHost::allocHost<int>(extents);
    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(int y = 0; y < size; ++y)
    {
        for(int x = 0; x < size; ++x)
        {
            if(y == 0 || y == (size - 1) || x == 0 || x == (size - 1))
            {
                CHECK(hostOutput[Vec{y, x}] == 0);
            }
            else
            {
                CHECK(hostOutput[Vec{y, x}] == 4);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - iota", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 8;
    Vec extents(size);

    // BEGIN-TUTORIAL-iota
    auto outputBuffer = onHost::alloc<int>(device, extents);
    onHost::iota<int>(queue, exec, 10, outputBuffer);
    // END-TUTORIAL-iota

    auto hostOutput = onHost::allocHost<int>(extents);
    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);
    for(size_t i = 0; i < size; ++i)
    {
        REQUIRE(hostOutput[i] == static_cast<int>(i) + 10);
    }
}

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - reduce", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 8;
    Vec extents(size);

    std::array<int, size> data{1, 2, 3, 4, 5, 6, 7, 8};

    // BEGIN-TUTORIAL-reduce
    auto inputBuffer = onHost::alloc<int>(device, extents);
    onHost::memcpy(queue, inputBuffer, data);

    auto reduceBuffer = onHost::alloc<int>(device, Vec{1u});
    int neutralElement = 0;
    onHost::reduce(queue, exec, neutralElement, reduceBuffer, std::plus{}, inputBuffer);
    // END-TUTORIAL-reduce

    auto hostOutput = onHost::allocHost<int>(size_t{1});
    onHost::memcpy(queue, hostOutput, reduceBuffer);
    onHost::wait(queue);

    REQUIRE(hostOutput[0] == 36);
}

// BEGIN-TUTORIAL-transformReduceFunctor
struct MultiplyValues
{
    ALPAKA_FN_ACC auto operator()(concepts::Simd auto const& a, concepts::Simd auto const& b) const
    {
        return a * b;
    }
};

// END-TUTORIAL-transformReduceFunctor

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - tranformReduce", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 8;
    Vec extents(size);

    std::array<int, size> data{1, 2, 3, 4, 5, 6, 7, 8};

    auto inputBuffer1 = onHost::alloc<int>(device, extents);
    auto inputBuffer2 = onHost::alloc<int>(device, extents);
    onHost::memcpy(queue, inputBuffer1, data);
    onHost::memcpy(queue, inputBuffer2, data);

    // BEGIN-TUTORIAL-transformReduceCall
    auto resultBuffer = onHost::alloc<int>(device, Vec{1u});
    onHost::transformReduce(
        queue,
        exec,
        // neutral element
        0,
        resultBuffer,
        // reduce functor
        std::plus{},
        // transform functor
        MultiplyValues{},
        // At least one input is required, we use two
        inputBuffer1,
        inputBuffer2);
    // END-TUTORIAL-transformReduceCall

    auto hostOutput = onHost::allocHost<int>(size_t{1});
    onHost::memcpy(queue, hostOutput, resultBuffer);
    onHost::wait(queue);

    REQUIRE(hostOutput[0] == 204);
}

// BEGIN-TUTORIAL-concurrentFunctor
struct MultiOutput
{
    ALPAKA_FN_ACC void operator()(concepts::SimdPtr auto a, concepts::SimdPtr auto const& b, concepts::SimdPtr auto c)
        const
    {
        a = a.load() + b.load();
        c = b.load() * b.load();
    }
};

// END-TUTORIAL-concurrentFunctor

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - concurrent", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 8;
    Vec extents(size);

    std::array<int, size> data{1, 2, 3, 4, 5, 6, 7, 8};

    // BEGIN-TUTORIAL-concurrentCall
    auto bufferA = onHost::alloc<int>(device, extents);
    auto bufferB = onHost::alloc<int>(device, extents);
    auto bufferC = onHost::alloc<int>(device, extents);
    onHost::memcpy(queue, bufferA, data);
    onHost::memcpy(queue, bufferB, data);
    onHost::fill(queue, bufferC, 0);

    onHost::concurrent<int>(queue, exec, bufferA.getExtents(), MultiOutput{}, bufferA, bufferB, bufferC);
    // END-TUTORIAL-concurrentCall

    auto hostOutputA = onHost::allocHost<int>(extents);
    auto hostOutputC = onHost::allocHost<int>(extents);
    onHost::memcpy(queue, hostOutputA, bufferA);
    onHost::memcpy(queue, hostOutputC, bufferC);
    onHost::wait(queue);

    for(size_t i = 0; i < size; ++i)
    {
        int const v = static_cast<int>(i) + 1;
        CHECK(hostOutputA[i] == v * 2);
        CHECK(hostOutputC[i] == v * v);
    }
}

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - scan", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 8;
    Vec extents(size);

    std::array<int, size> data{1, 2, 3, 4, 5, 6, 7, 8};

    auto inputBuffer = onHost::alloc<int>(device, extents);
    onHost::memcpy(queue, inputBuffer, data);

    // BEGIN-TUTORIAL-scan
    auto outputBuffer = onHost::alloc<int>(device, extents);
    auto tmpBuffer = onHost::alloc<std::byte>(device, onHost::getScanBufferSize<int>(inputBuffer.getExtents()));
    onHost::inclusiveScan(queue, exec, tmpBuffer, outputBuffer, inputBuffer);
    // END-TUTORIAL-scan

    auto hostOutput = onHost::allocHost<int>(extents);
    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(hostOutput[0] == 1);
    CHECK(hostOutput[1] == 3);
    CHECK(hostOutput[2] == 6);
    CHECK(hostOutput[3] == 10);
    CHECK(hostOutput[4] == 15);
    CHECK(hostOutput[5] == 21);
    CHECK(hostOutput[6] == 28);
    CHECK(hostOutput[7] == 36);
}

// BEGIN-TUTORIAL-generatorFunctor
struct AddLinearIdx
{
    ALPAKA_FN_ACC auto operator()(int const& value, size_t const& linearIdx) const -> int
    {
        return value + static_cast<int>(linearIdx);
    }
};

// END-TUTORIAL-generatorFunctor

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - linear generator", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr size_t size = 8;
    Vec extents(size);

    std::array<int, size> data{1, 2, 3, 4, 5, 6, 7, 8};
    auto inputBuffer = onHost::alloc<int>(device, extents);
    onHost::memcpy(queue, inputBuffer, data);

    // BEGIN-TUTORIAL-generatorCall
    auto generator = LinearizedIdxGenerator{extents};
    auto outputBuffer = onHost::alloc<int>(device, extents);
    onHost::transform(queue, exec, outputBuffer, ScalarFunc{AddLinearIdx{}}, inputBuffer, generator);
    // END-TUTORIAL-generatorCall

    auto hostOutput = onHost::allocHost<int>(extents);
    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < size; ++i)
    {
        CHECK(hostOutput[i] == (data[i] + static_cast<int>(generator[i])));
    }
}
