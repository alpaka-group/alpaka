/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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

#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>


//#############################################################################
class BlockSyncPredicateTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        using Idx = alpaka::idx::Idx<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
        auto const blockThreadExtent(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
        auto const blockThreadIdx1D(alpaka::idx::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u]);
        auto const blockThreadExtent1D(blockThreadExtent.prod());

        // syncBlockThreadsPredicate<alpaka::block::sync::op::Count>
        {
            Idx const modulus(2u);
            int const predicate(static_cast<int>(blockThreadIdx1D % modulus));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::Count>(acc, predicate));
            auto const expectedResult(static_cast<int>(blockThreadExtent1D / modulus));
            ALPAKA_CHECK(*success, expectedResult == result);
        }
        {
            Idx const modulus(3u);
            int const predicate(static_cast<int>(blockThreadIdx1D % modulus));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::Count>(acc, predicate));
            auto const expectedResult(static_cast<int>(blockThreadExtent1D - ((blockThreadExtent1D + modulus - static_cast<Idx>(1u)) / modulus)));
            ALPAKA_CHECK(*success, expectedResult == result);
        }

        // syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>
        {
            int const predicate(1);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate(0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            int const predicate(blockThreadIdx1D != 0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            ALPAKA_CHECK(*success, result == 0);
        }

        // syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>
        {
            int const predicate(1);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate(0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            int const predicate(static_cast<int>(blockThreadIdx1D != 1));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            ALPAKA_CHECK(*success, result == 1);
        }
    }
};

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    BlockSyncPredicateTestKernel kernel;

    // 4^Dim
    {
        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(4u)));

        REQUIRE(
            fixture(
                kernel));
    }

    // 1^Dim
    {
        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        REQUIRE(
            fixture(
                kernel));
    }
}
};

TEST_CASE( "synchronizePredicate", "[blockSync]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}
