/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// NVCC needs --expt-extended-lambda
#if !defined(__NVCC__) || \
    ( defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__) )


#include <alpaka/core/BindScope.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

//-----------------------------------------------------------------------------
struct TestTemplateBindScope
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg = 42u;

    REQUIRE(
        fixture(
            alpaka::core::bindScope< alpaka::core::Scope::HostDevice > (
                ALPAKA_FN_LAMBDA
                (TAcc const & acc,
                    bool * success,
                    std::uint32_t const & arg1)
                -> void
                {
                    alpaka::ignore_unused(acc);

                    ALPAKA_CHECK(*success, 42u == arg1);
                }), arg));
}
};

struct TestTemplateBindScopeShortcut
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg = 42u;

    REQUIRE(
        fixture(
            ALPAKA_FN_SCOPE_HOST_DEVICE(
                ALPAKA_FN_LAMBDA
                (TAcc const & acc,
                    bool * success,
                    std::uint32_t const & arg1)
                -> void
                {
                    alpaka::ignore_unused(acc);

                    ALPAKA_CHECK(*success, 42u == arg1);
                }), arg));
}
};

TEST_CASE( "bindScopedLambdaKernelIsWorking", "[kernel]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt<1u>,
        std::size_t>;
    alpaka::meta::forEachType< TestAccs >( TestTemplateBindScope() );
}

TEST_CASE( "bindScopeShortcutLambdaKernelIsWorking", "[kernel]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt<1u>,
        std::size_t>;
    alpaka::meta::forEachType< TestAccs >( TestTemplateBindScopeShortcut() );
}

#endif
