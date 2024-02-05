/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using Elem = std::uint32_t;
using Dim = alpaka::DimInt<2u>;
using Idx = std::uint32_t;

// These forward declarations are only necessary when you want to access those variables
// from a different compilation unit and should be moved to a common header.
// Here they are used to silence clang`s -Wmissing-variable-declarations warning
// that forces every non-static variable to be declared with extern before the are defined.
// EXTERN_ALPAKA_STATIC_ACC_MEM_GLOBAL(Elem[3][2], g_globalMemory2DUninitialized);
//EXTERN_ALPAKA_STATIC_ACC_MEM_GLOBAL(Elem[3][2], g_globalMemory2DUninitialized);
ALPAKA_STATIC_ACC_MEM_GLOBAL(Elem[3][2], g_globalMemory2DUninitialized);

// These forward declarations are only necessary when you want to access those variables
// from a different compilation unit and should be moved to a common header.
// Here they are used to silence clang`s -Wmissing-variable-declarations warning
// that forces every non-static variable to be declared with extern before the are defined.
// EXTERN_ALPAKA_STATIC_ACC_MEM_CONSTANT(Elem[3][2], g_constantMemory2DUninitialized);
ALPAKA_STATIC_ACC_MEM_CONSTANT(Elem[3][2], g_constantMemory2DUninitialized);

//! Uses static device memory on the accelerator defined globally for the whole compilation unit.
struct StaticDeviceMemoryTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    // ALPAKA_FN_ACC void operator()(TAcc const& acc, bool* success) const
    ALPAKA_FN_ACC void operator()(TAcc const& acc, bool* success) const
    {
        auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        auto const offset = gridThreadExtent[1u] * gridThreadIdx[0u] + gridThreadIdx[1u];
        auto const val = offset;

        ALPAKA_CHECK(*success, val == *((&g_globalMemory2DUninitialized<TAcc>.get())[0][0] + offset));
        printf(
            "%u = %u\n",
            val,
            *((&g_globalMemory2DUninitialized<TAcc>.get())[0][0] + offset)); // compila anche senza get()
    }
};

struct ConstantDeviceMemoryTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, bool* success) const
    {
        auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        auto const offset = gridThreadExtent[1u] * gridThreadIdx[0u] + gridThreadIdx[1u];
        auto const val = offset;

        ALPAKA_CHECK(*success, val == *((&g_constantMemory2DUninitialized<TAcc>.get())[0][0] + offset));
        printf(
            "%u = %u\n",
            val,
            *((&g_constantMemory2DUninitialized<TAcc>.get())[0][0] + offset)); // compila anche senza get()
    }
};

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_TARGET_CPU)
template<typename TDim, typename TIdx>
using EnabledAccsElseInt = std::tuple<alpaka::AccCpuSycl<TDim, TIdx>>;
template<typename TDim, typename TIdx>
using EnabledAccs = typename alpaka::meta::Filter<EnabledAccsElseInt<TDim, TIdx>, std::is_class>;
using TestAccs = EnabledAccs<Dim, Idx>;
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_TARGET_GPU)
template<typename TDim, typename TIdx>
using EnabledAccsElseInt = std::tuple<alpaka::AccGpuSyclIntel<TDim, TIdx>>;
template<typename TDim, typename TIdx>
using EnabledAccs = typename alpaka::meta::Filter<EnabledAccsElseInt<TDim, TIdx>, std::is_class>;
using TestAccs = EnabledAccs<Dim, Idx>;
#else
using TestAccs = alpaka::test::EnabledAccs<Dim, Idx>;
#endif

TEMPLATE_LIST_TEST_CASE("staticDeviceMemoryGlobal", "[viewStaticAccMem]", TestAccs)
{
    using Acc = TestType;
    using DevAcc = alpaka::Dev<Acc>;

    auto const platformAcc = alpaka::Platform<Acc>{};

    std::cout << " acc:" << alpaka::core::demangled<Acc> << std::endl;
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    alpaka::Vec<Dim, Idx> const extent(3u, 2u);

    alpaka::test::KernelExecutionFixture<Acc> fixture(extent);

    StaticDeviceMemoryTestKernel kernel;

    // uninitialized static global device memory
    {
        auto const platformHost = alpaka::PlatformCpu{};
        auto const devHost = alpaka::getDevByIdx(platformHost, 0);

        using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;
        QueueAcc queueAcc(devAcc);

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        auto bufHost = alpaka::createView(devHost, data.data(), extent);

        alpaka::memcpy(queueAcc, g_globalMemory2DUninitialized<Acc>, bufHost, extent);
        alpaka::wait(queueAcc);

        REQUIRE(fixture(kernel));

        std::vector<Elem> data2(6, 0u);
        auto bufHost2 = alpaka::createView(devHost, data2.data(), extent);
        alpaka::memcpy(queueAcc, bufHost2, g_globalMemory2DUninitialized<Acc>, extent);
        alpaka::wait(queueAcc);
        REQUIRE(data == data2);
    }
}

TEMPLATE_LIST_TEST_CASE("staticDeviceMemoryConstant", "[viewStaticAccMem]", TestAccs)
{
    using Acc = TestType;
    using DevAcc = alpaka::Dev<Acc>;

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    alpaka::Vec<Dim, Idx> const extent(3u, 2u);

    alpaka::test::KernelExecutionFixture<Acc> fixture(extent);

    ConstantDeviceMemoryTestKernel kernel;

    // uninitialized static constant device memory
    {
        auto const platformHost = alpaka::PlatformCpu{};
        auto const devHost = alpaka::getDevByIdx(platformHost, 0);

        using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;
        QueueAcc queueAcc(devAcc);

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        auto bufHost = alpaka::createView(devHost, data.data(), extent);

        alpaka::memcpy(queueAcc, g_constantMemory2DUninitialized<Acc>, bufHost);
        alpaka::wait(queueAcc);

        REQUIRE(fixture(kernel));

        std::vector<Elem> data2(6, 0u);
        auto bufHost2 = alpaka::createView(devHost, data2.data(), extent);
        alpaka::memcpy(queueAcc, bufHost2, g_constantMemory2DUninitialized<Acc>);
        alpaka::wait(queueAcc);
        REQUIRE(data == data2);
    }
}
