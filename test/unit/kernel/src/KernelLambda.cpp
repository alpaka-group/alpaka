/**
 * \file
 * Copyright 2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// NVCC needs --expt-extended-lambda
#if !defined(__NVCC__) || \
    ( defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__) )

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <catch2/catch.hpp>


//-----------------------------------------------------------------------------
struct TestTemplateLambda
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    auto kernel =
        [] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success)
        -> void
        {
            ALPAKA_CHECK(
                *success,
                static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
        };

    REQUIRE(fixture(kernel));
}
};

//-----------------------------------------------------------------------------
struct TestTemplateArg
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg = 42u;
    auto kernel =
        [] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success,
            std::uint32_t const & arg1)
        -> void
        {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(*success, 42u == arg1);
        };

    REQUIRE(fixture(kernel, arg));
}
};

//-----------------------------------------------------------------------------
struct TestTemplateCapture
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg = 42u;

#if BOOST_COMP_CLANG >= BOOST_VERSION_NUMBER(5,0,0)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-lambda-capture"
#endif
    auto kernel =
        [arg] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success)
        -> void
        {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(*success, 42u == arg);
        };
#if BOOST_COMP_CLANG >= BOOST_VERSION_NUMBER(5,0,0)
    #pragma clang diagnostic pop
#endif

    REQUIRE(fixture(kernel));
}
};


TEST_CASE( "lambdaKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateLambda() );
}

TEST_CASE( "lambdaKernelWithArgumentIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateArg() );
}

TEST_CASE( "lambdaKernelWithCapturingIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateCapture() );
}

#endif
