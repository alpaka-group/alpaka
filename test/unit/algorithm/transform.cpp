/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Stencil SIMD functor
 *
 * The stencil functor is getting a SimdPtr as input which allows to call the operator[] to change the index.
 * In this test we do not shift the memory to avoid out of memory access.
 * We cannot use generic lambdas with CUDA therefore we need to write a functor
 */
struct StencilAdd
{
    constexpr auto operator()(concepts::SimdPtr auto const& a, concepts::SimdPtr auto const& b) const
    {
        return a.load() + b.load();
    }
};

struct StencilAddWithAcc
{
    constexpr auto operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto const& a,
        concepts::SimdPtr auto const& b) const
    {
        return a.load() + b.load();
    }
};

struct AddUpCastWithAcc
{
    constexpr auto operator()(
        onAcc::concepts::Acc auto const&,
        concepts::Simd auto const& a,
        concepts::Simd auto const& b) const
    {
        return pCast<size_t>(a + b);
    }
};

struct ScalarOpWithAcc
{
    constexpr auto operator()(onAcc::concepts::Acc auto const&, auto const& a, auto const& b) const
    {
        using ValueType = GetValueType_t<ALPAKA_TYPEOF(a)>;
        // The algorithm is using SIMD internally, forwarding single lanes to this functor can result in forwarding
        // reference wrappers instead of values. By calling operator + we enforce casting to the value type.
        return math::min(a + ValueType{1}, +b);
    }
};

struct TestWithMdSpan
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO("run func : " << onHost::demangledName(std::get<1>(setup)));

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = decltype(std::get<1>(setup)(std::declval<DataType>(), std::declval<DataType>()));
        onHost::SharedBuffer computeBufferOut = onHost::allocDeferred<OutDataType>(computeQueue, extentMd);
        onHost::SharedBuffer computeBufferIn0 = onHost::allocDeferred<DataType>(computeQueue, extentMd);
        onHost::SharedBuffer computeBufferIn1 = onHost::allocLikeDeferred(computeQueue, computeBufferIn0);
        onHost::SharedBuffer hostBufferIota = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocLike(onHost::makeHostDevice(), computeBufferOut);

        // initialize with the linearized index
        DataType iotaCounter = 0;
        for(auto& value : hostBufferIota)
        {
            value = iotaCounter;
            ++iotaCounter;
        }

        onHost::memcpy(computeQueue, computeBufferIn0, hostBufferIota);
        onHost::memcpy(computeQueue, computeBufferIn1, hostBufferIota);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::transform(
            computeQueue,
            exec,
            computeBufferOut,
            std::get<0>(setup),
            computeBufferIn0,
            computeBufferIn1);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for transform: " << std::chrono::duration<double>(endT - beginT).count()
                                   << "s data size: " << computeBufferOut.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut[idx] == std::get<1>(setup)(refIotaCounter, refIotaCounter));
                ++refIotaCounter;
            });
    }
};

template<typename T_DataType>
void prepareTest(auto cfg, concepts::Vector auto extentMd, auto const& setupTuple)
{
    using DataType = T_DataType;
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue computeQueue = computeDev.makeQueue();

    // execute for each functor
    std::apply(
        [&](auto const&... setup)
        { (std::get<2>(setup).template executeTest<DataType>(exec, computeQueue, setup, extentMd), ...); },
        setupTuple);
}

TEMPLATE_LIST_TEST_CASE("transform", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto functorList = std::make_tuple(
        std::make_tuple(std::minus{}, std::minus{}, TestWithMdSpan{}),
        std::make_tuple(std::plus{}, std::plus{}, TestWithMdSpan{}),
        // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
        std::make_tuple(StencilFunc{StencilAdd{}}, std::plus{}, TestWithMdSpan{}),
        std::make_tuple(StencilFunc{StencilAddWithAcc{}}, std::plus{}, TestWithMdSpan{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrap the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                       { return math::min(a + DataType{1}, b); }},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithMdSpan{}),
        std::make_tuple(
            ScalarFunc{ScalarOpWithAcc{}},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithMdSpan{}),
        // different output type
        std::make_tuple(
            AddUpCastWithAcc{},
            [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); },
            TestWithMdSpan{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, functorList), ...); }, extentMdList);
}

struct TestWithGenerator
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO("run func : " << onHost::demangledName(std::get<1>(setup)));

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = decltype(std::get<1>(setup)(std::declval<DataType>(), std::declval<DataType>()));
        onHost::SharedBuffer computeBufferOut = onHost::allocDeferred<OutDataType>(computeQueue, extentMd);
        onHost::SharedBuffer computeBufferIn0 = onHost::allocDeferred<DataType>(computeQueue, extentMd);
        auto generator = LinearizedIdxGenerator{extentMd};
        onHost::SharedBuffer hostBufferIota = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocLike(onHost::makeHostDevice(), computeBufferOut);

        // initialize with the linearized index
        DataType iotaCounter = 0;
        for(auto& value : hostBufferIota)
        {
            value = iotaCounter;
            ++iotaCounter;
        }

        onHost::memcpy(computeQueue, computeBufferIn0, hostBufferIota);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::transform(computeQueue, exec, computeBufferOut, std::get<0>(setup), computeBufferIn0, generator);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for transform: " << std::chrono::duration<double>(endT - beginT).count()
                                   << "s data size: " << computeBufferOut.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut[idx] == std::get<1>(setup)(refIotaCounter, generator[idx]));
                ++refIotaCounter;
            });
    }
};

TEMPLATE_LIST_TEST_CASE("transform generator", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto functorList = std::make_tuple(
        std::make_tuple(std::minus{}, std::minus{}, TestWithGenerator{}),
        std::make_tuple(std::plus{}, std::plus{}, TestWithGenerator{}),
        // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
        std::make_tuple(StencilFunc{StencilAdd{}}, std::plus{}, TestWithGenerator{}),
        std::make_tuple(StencilFunc{StencilAddWithAcc{}}, std::plus{}, TestWithGenerator{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrap the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                       { return math::min(a + DataType{1}, b); }},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithGenerator{}),
        std::make_tuple(
            ScalarFunc{ScalarOpWithAcc{}},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithGenerator{}),
        // different output type
        std::make_tuple(
            AddUpCastWithAcc{},
            [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); },
            TestWithGenerator{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, functorList), ...); }, extentMdList);
}

/** Cross stencil
 *
 * N-dimensional asymmetric cross stencil. The positive direction is weighted with 5 and the negative with 3.
 * The middle value is not weighted.
 */
struct CrossStencil
{
    constexpr auto operator()(concepts::SimdPtr auto const& in) const
    {
        using SimdPtrType = ALPAKA_TYPEOF(in);
        using VecIdxType = typename SimdPtrType::IdxType;

        // Load the value the simd pointer is pointing to.
        concepts::Simd auto result = in.load();

        // Arbitrary constant added to the result.
        using SimdType = ALPAKA_TYPEOF(result);
        concepts::Simd auto const value = SimdType::fill(42);
        result += value;

        for(uint32_t d = 0u; d < VecIdxType::dim(); ++d)
        {
            /* Shift relative to the current element pointed to into the negative or positive direction within a
             * dimension.
             */
            concepts::Vector auto negative = VecIdxType::fill(0);
            concepts::Vector auto positive = VecIdxType::fill(0);
            negative[d] = -1;
            positive[d] = 1;

            result += in[negative].load() * 3;
            result += in[positive].load() * 5;
        }

        return result;
    }
};

template<typename T_DataType, typename T_StencilFn>
void testStencilTransform(auto cfg, concepts::Vector auto extentMd)
{
    using DataType = T_DataType;
    using Extent = std::decay_t<decltype(extentMd)>;

    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue computeQueue = computeDev.makeQueue();

    INFO("extents    : " << extentMd);

    auto const halo = Extent::fill(1);
    auto const innerExtent = extentMd - Extent::fill(2);

    onHost::SharedBuffer computeIn = onHost::allocDeferred<DataType>(computeQueue, extentMd);
    // The output does not need helo's
    onHost::SharedBuffer computeOut = onHost::allocDeferred<DataType>(computeQueue, innerExtent);

    onHost::SharedBuffer hostIn = onHost::allocLike(onHost::makeHostDevice(), computeIn);
    onHost::SharedBuffer hostOut = onHost::allocLike(onHost::makeHostDevice(), computeOut);

    meta::ndLoopIncIdx(
        extentMd,
        [&](auto idx)
        {
            /* Use a fake size of 1000 for each direction, this would make the index human.
             * Accidentally switching the index order z,y,x in SImdPtr during the stencil access will be detracted
             * and easy to debug. e.g. (y,x) -> (13,5) will become 13005
             */
            int32_t value = alpaka::linearize(ALPAKA_TYPEOF(extentMd)::fill(1000), idx);

            hostIn[idx] = value;
        });

    onHost::memcpy(computeQueue, computeIn, hostIn);

    /* For the stencil operation we work only on the inner volume, this allows performing the stencil operation
     * without handling out of memory access at the boundaries.
     */
    auto shiftedComputeIn = computeIn.getView().getSubView(halo, innerExtent);

    onHost::transform(computeQueue, exec, computeOut, StencilFunc{T_StencilFn{}}, shiftedComputeIn);
    onHost::memcpy(computeQueue, hostOut, computeOut);
    onHost::wait(computeQueue);

    meta::ndLoopIncIdx(
        innerExtent,
        [&](concepts::Vector auto outIdx)
        {
            concepts::Vector auto const inIdx = outIdx + halo;

            DataType expected = hostIn[inIdx] + 42;

            for(uint32_t d = 0u; d < extentMd.dim(); ++d)
            {
                // absolut offset from the origin of hostIn including helo
                auto negative = inIdx;
                auto positive = inIdx;

                --negative[d];
                ++positive[d];

                expected += hostIn[negative] * 3;
                expected += hostIn[positive] * 5;
            }

            REQUIRE(hostOut[outIdx] == expected);
        });
}

/** Validate the transform stencil feature.
 *
 * Additionally, this checks take care that SimdPtr::operator[]() works as expected.
 */
TEMPLATE_LIST_TEST_CASE("transform SimdPtr cross stencil", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int32_t;

    auto extentMdList = std::make_tuple(Vec{17}, Vec{17, 19}, Vec{13, 17, 19}, Vec{7, 11, 13, 17});

    std::apply(
        [&](auto... extents) { (testStencilTransform<DataType, CrossStencil>(cfg, extents), ...); },
        extentMdList);
}
