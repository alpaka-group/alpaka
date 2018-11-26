/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <catch2/catch.hpp>

#include <functional>

//-----------------------------------------------------------------------------
template<
    typename TAcc>
void ALPAKA_FN_ACC kernelFn(
    TAcc const & acc,
    bool * success,
    std::int32_t val)
{
    alpaka::ignore_unused(acc);

    ALPAKA_CHECK(*success, 42 == val);
}

// std::function and std::bind is only allowed on CPU
#if !BOOST_LANG_CUDA && !BOOST_LANG_HIP
//-----------------------------------------------------------------------------
struct TestTemplateStdFunction
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    const auto kernel = std::function<void(TAcc const &, bool *, std::int32_t)>( kernelFn<TAcc> );
    REQUIRE(fixture(kernel, 42));
  }
};

TEST_CASE( "stdFunctionKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateStdFunction() );
}

//-----------------------------------------------------------------------------
struct TestTemplateStdBind
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    const auto kernel = std::bind( kernelFn<TAcc>, std::placeholders::_1, std::placeholders::_2, 42 );
    REQUIRE(fixture(kernel));
  }
};

TEST_CASE( "stdBindKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateStdBind() );
}
#endif
