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
        return a * ValueType{2} + b;
    }
};

namespace myTest
{
    // This functor does not support Simd and must be wrapped by ScalarFunc
    struct MinValue
    {
        constexpr auto operator()(auto const& a, auto const& b) const
        {
            return math::min(a, b);
        }
    };

    ALPAKA_FN_ACC void alpakaAtomicInvoke(
        MinValue const&,
        alpaka::onAcc::concepts::Acc auto const& acc,
        auto* dest,
        auto const src)
    {
        onAcc::atomicMin(acc, dest, src);
    }
} // namespace myTest

struct TestWithMdSpan
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO(
            "run reduce(func) : " << onHost::demangledName(std::get<1>(setup).second) << "("
                                  << onHost::demangledName(std::get<2>(setup).second) << ")");

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = ALPAKA_TYPEOF(std::get<0>(setup));
        onHost::SharedBuffer computeBufferOut = onHost::alloc<OutDataType>(computeDev, extentMd.fill(1));
        onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
        onHost::SharedBuffer computeBufferIn1 = onHost::allocLike(computeDev, computeBufferIn0);
        onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

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

        onHost::transformReduce(
            computeQueue,
            exec,
            std::get<0>(setup),
            computeBufferOut,
            std::get<1>(setup).first,
            std::get<2>(setup).first,
            computeBufferIn0,
            computeBufferIn1);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for transform reduce: " << std::chrono::duration<double>(endT - beginT).count()
                                          << "s data size: " << hostBufferIota.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        auto result = std::get<0>(setup);
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                alpaka::unused(idx);
                result = std::get<1>(setup).second(result, std::get<2>(setup).second(refIotaCounter, refIotaCounter));
                ++refIotaCounter;
            });
        CHECK(*hostBufferOut.data() == result);
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
        { (std::get<3>(setup).template executeTest<DataType>(exec, computeQueue, setup, extentMd), ...); },
        setupTuple);
}

TEMPLATE_LIST_TEST_CASE("transformReduce", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto setups = std::make_tuple(
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(std::plus{}, std::plus{}),
            TestWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAdd{}}, std::plus{}),
            TestWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAddWithAcc{}}, std::plus{}),
            TestWithMdSpan{}),
        std::make_tuple(
            size_t{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                AddUpCastWithAcc{},
                [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); }),
            TestWithMdSpan{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrap the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                           {
                               /* Temporary variable is required to avoid icpx 2026.0 linker issue.
                                * Running pass
                                * "function(vpo-cfg-restructuring,replace-with-math-library-functions,inject-tli-mappings,auto-vec-dir-insert,vplan-vec,replace-with-math-library-functions)"
                                * on module "main" llvm-foreach: Segmentation fault (core dumped)
                                */
                               auto lhs = a + DataType{1};
                               return alpaka::math::min(lhs, b);
                           }},
                [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); }),
            TestWithMdSpan{}),
        std::make_tuple(
            std::numeric_limits<DataType>::max(),
            std::make_pair(ScalarFunc{myTest::MinValue{}}, myTest::MinValue{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(std::plus{}, std::plus{}),
            TestWithMdSpan{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
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
        INFO(
            "run reduce(func) : " << onHost::demangledName(std::get<1>(setup).second) << "("
                                  << onHost::demangledName(std::get<2>(setup).second) << ")");

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = ALPAKA_TYPEOF(std::get<0>(setup));
        onHost::SharedBuffer computeBufferOut = onHost::alloc<OutDataType>(computeDev, extentMd.fill(1));
        onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
        auto generator = LinearizedIdxGenerator{extentMd};
        onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

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

        onHost::transformReduce(
            computeQueue,
            exec,
            std::get<0>(setup),
            computeBufferOut,
            std::get<1>(setup).first,
            std::get<2>(setup).first,
            computeBufferIn0,
            generator);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for transform reduce: " << std::chrono::duration<double>(endT - beginT).count()
                                          << "s data size: " << hostBufferIota.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        auto result = std::get<0>(setup);
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                result = std::get<1>(setup).second(result, std::get<2>(setup).second(refIotaCounter, generator[idx]));
                ++refIotaCounter;
            });
        CHECK(*hostBufferOut.data() == result);
    }
};

TEMPLATE_LIST_TEST_CASE("transformReduce generator", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto setups = std::make_tuple(
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(std::plus{}, std::plus{}),
            TestWithGenerator{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAdd{}}, std::plus{}),
            TestWithGenerator{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAddWithAcc{}}, std::plus{}),
            TestWithGenerator{}),
        std::make_tuple(
            size_t{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                AddUpCastWithAcc{},
                [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); }),
            TestWithGenerator{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrap the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                           {
                               /* Temporary variable is required to avoid icpx 2026.0 linker issue.
                                * Running pass
                                * "function(vpo-cfg-restructuring,replace-with-math-library-functions,inject-tli-mappings,auto-vec-dir-insert,vplan-vec,replace-with-math-library-functions)"
                                * on module "main" llvm-foreach: Segmentation fault (core dumped)
                                */
                               auto lhs = a + DataType{1};
                               return math::min(lhs, b);
                           }},
                [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); }),
            TestWithGenerator{}),
        std::make_tuple(
            std::numeric_limits<DataType>::max(),
            std::make_pair(ScalarFunc{myTest::MinValue{}}, myTest::MinValue{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(std::plus{}, std::plus{}),
            TestWithGenerator{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
}

struct OnAccTransformReduceKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto out,
        auto const& extentMd,
        auto const& neutralElement,
        auto const& reduceFunc,
        auto const& transformFunc,
        auto const&... inputs) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::WorkerGroup{extentMd.fill(0), extentMd.fill(1)}};
        out[0u] = simdGrid.transformReduce(acc, extentMd, neutralElement, reduceFunc, transformFunc, inputs...);
    }
};

struct TestOnAccWithMdSpan
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO(
            "run onAcc reduce(func) : " << onHost::demangledName(std::get<1>(setup).second) << "("
                                        << onHost::demangledName(std::get<2>(setup).second) << ")");

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = ALPAKA_TYPEOF(std::get<0>(setup));
        onHost::SharedBuffer computeBufferOut = onHost::alloc<OutDataType>(computeDev, 1u);
        onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
        onHost::SharedBuffer computeBufferIn1 = onHost::allocLike(computeDev, computeBufferIn0);
        onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

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

        computeQueue.enqueue(
            onHost::FrameSpec{1u, 1u, exec},
            KernelBundle{
                OnAccTransformReduceKernel{},
                computeBufferOut,
                extentMd,
                std::get<0>(setup),
                std::get<1>(setup).first,
                std::get<2>(setup).first,
                computeBufferIn0,
                computeBufferIn1});

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for onAcc transform reduce: " << std::chrono::duration<double>(endT - beginT).count()
                                                << "s data size: " << hostBufferIota.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        DataType refIotaCounter = 0;
        auto result = std::get<0>(setup);
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                alpaka::unused(idx);
                result = std::get<1>(setup).second(result, std::get<2>(setup).second(refIotaCounter, refIotaCounter));
                ++refIotaCounter;
            });
        CHECK(*hostBufferOut.data() == result);
    }
};

template<typename T_DataType>
void prepareOnAccTest(auto cfg, concepts::Vector auto extentMd, auto const& setupTuple)
{
    using DataType = T_DataType;

    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue computeQueue = computeDev.makeQueue();

    std::apply(
        [&](auto const&... setup)
        { (std::get<3>(setup).template executeTest<DataType>(exec, computeQueue, setup, extentMd), ...); },
        setupTuple);
}

TEMPLATE_LIST_TEST_CASE("onAcc transformReduce", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    auto setups = std::make_tuple(
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(std::plus{}, std::plus{}),
            TestOnAccWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(StencilFunc{StencilAdd{}}, std::plus{}),
            TestOnAccWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(StencilFunc{StencilAddWithAcc{}}, std::plus{}),
            TestOnAccWithMdSpan{}),
        std::make_tuple(
            size_t{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                AddUpCastWithAcc{},
                [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); }),
            TestOnAccWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                           {
                               /* std::min() instead of alpaka::min() is used to avoid icpx 2026.0 linker issue.
                                * Running pass
                                * "function(vpo-cfg-restructuring,replace-with-math-library-functions,inject-tli-mappings,auto-vec-dir-insert,vplan-vec,replace-with-math-library-functions)"
                                * on module "main" llvm-foreach: Segmentation fault (core dumped)
                                */
                               return std::min(a + DataType{1}, b);
                           }},
                [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); }),
            TestOnAccWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                ScalarFunc{ScalarOpWithAcc{}},
                [](DataType const& a, DataType const& b) { return a * DataType{2} + b; }),
            TestOnAccWithMdSpan{}),
        std::make_tuple(
            std::numeric_limits<DataType>::max(),
            std::make_pair(ScalarFunc{myTest::MinValue{}}, myTest::MinValue{}),
            std::make_pair(std::plus{}, std::plus{}),
            TestOnAccWithMdSpan{}));

    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareOnAccTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
}
