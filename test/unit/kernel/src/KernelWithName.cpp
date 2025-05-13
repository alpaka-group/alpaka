/* Copyright 2025 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/kernel/Traits.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class KernelWithName
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success, std::int32_t val) const -> void
    {
        ALPAKA_CHECK(*success, 42 == val);
    }
};

ALPAKA_KERNEL_NAME(KernelWithName, kernelWithName)

TEMPLATE_LIST_TEST_CASE("KernelWithName", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelWithName kernel;

    REQUIRE(fixture(kernel, 42));
}

class KernelWithScopedName
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success, std::int32_t val) const -> void
    {
        ALPAKA_CHECK(*success, 42 == val);
    }
};

ALPAKA_KERNEL_SCOPED_NAME(KernelWithScopedName, scope, kernelWithName)

TEMPLATE_LIST_TEST_CASE("KernelWithScopedName", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelWithScopedName kernel;

    REQUIRE(fixture(kernel, 42));
}
