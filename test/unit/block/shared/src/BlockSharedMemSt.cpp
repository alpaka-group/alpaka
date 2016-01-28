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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>                  // alpaka::test::acc::TestAccs
#include <alpaka/test/stream/Stream.hpp>            // alpaka::test::stream::DefaultStream
#include <alpaka/test/KernelExecutionFixture.hpp>   // alpaka::test::KernelExecutionFixture

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(blockSharedMemSt)

                                                                                                                                #include <iostream>
//#############################################################################
//!
//#############################################################################
class BlockSharedMemStNonNullTestKernel
{
public:
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
        auto && a = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_REQUIRE_NE(static_cast<std::uint32_t *>(nullptr), &a);

        auto && b = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_REQUIRE_NE(static_cast<std::uint32_t *>(nullptr), &b);

        auto && c = alpaka::block::shared::st::allocVar<float, __COUNTER__>(acc);
        BOOST_REQUIRE_NE(static_cast<float *>(nullptr), &c);

        auto && d = alpaka::block::shared::st::allocVar<double, __COUNTER__>(acc);
        BOOST_REQUIRE_NE(static_cast<double *>(nullptr), &d);

        auto && e = alpaka::block::shared::st::allocVar<std::uint64_t, __COUNTER__>(acc);
        BOOST_REQUIRE_NE(static_cast<std::uint64_t *>(nullptr), &e);
    }
};

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    nonNull,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::Vec<Dim, Size>::ones());

    BlockSharedMemStNonNullTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

//#############################################################################
//!
//#############################################################################
class BlockSharedMemStSameTypeDifferentAdressTestKernel
{
public:
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
        auto && a = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        auto && b = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_REQUIRE_NE(&a, &b);
        auto && c = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_REQUIRE_NE(&b,&c);
    }
};

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    sameTypeDifferentAdress,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::Vec<Dim, Size>::ones());

    BlockSharedMemStSameTypeDifferentAdressTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()
