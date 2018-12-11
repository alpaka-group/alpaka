/**
 * \file
 * Copyright 2015-2018 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

// Generic lambdas are a C++14 feature.
#if !defined(BOOST_NO_CXX14_GENERIC_LAMBDAS)
// CUDA C Programming guide says: "__host__ __device__ extended lambdas cannot be generic lambdas"
// However, it seems to work on all compilers except MSVC even though it is documented differently.
#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_COMP_MSVC)
//-----------------------------------------------------------------------------
struct TestTemplateGeneric
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
            auto const & acc,
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
struct TestTemplateVariadic
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg1 = 42u;
    std::uint32_t const arg2 = 43u;
    auto kernel =
        [] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success,
            auto ... args)
        -> void
        {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(
                *success,
                alpaka::meta::foldr([](auto a, auto b){return a + b;}, args...) == (42u + 43u));
        };

    REQUIRE(fixture(kernel, arg1, arg2));
}
};

TEST_CASE( "genericLambdaKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateGeneric() );
}

TEST_CASE( "variadicGenericLambdaKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateVariadic() );
}

#endif
#endif
