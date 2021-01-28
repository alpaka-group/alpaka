/* Copyright 2019-2021 Axel Huebl, Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/kernel/Traits.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

class KernelWithConstructorAndMember
{
public:
    ALPAKA_FN_HOST KernelWithConstructorAndMember(std::int32_t const val = 42) : m_val(val)
    {
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TMemoryHandle>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::experimental::
            Accessor<TMemoryHandle, bool, alpaka::Idx<TAcc>, 1, alpaka::experimental::WriteAccess> const success) const
        -> void
    {
        alpaka::ignore_unused(acc);

        ALPAKA_CHECK(success[0], 42 == m_val);
    }

private:
    std::int32_t m_val;
};

TEMPLATE_LIST_TEST_CASE("kernelWithConstructorAndMember", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelWithConstructorAndMember kernel(42);

    REQUIRE(fixture(kernel));
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstructorDefaultParamAndMember", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelWithConstructorAndMember kernel;

    REQUIRE(fixture(kernel));
}
