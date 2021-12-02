/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/block/sync/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

class BlockSyncPredicateTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const block_thread_idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const block_thread_extent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const block_thread_idx1_d = alpaka::mapIdx<1u>(block_thread_idx, block_thread_extent)[0u];
        auto const block_thread_extent1_d = block_thread_extent.prod();

        // syncBlockThreadsPredicate<alpaka::BlockCount>
        {
            Idx const modulus = 2u;
            auto const predicate = static_cast<int>(block_thread_idx1_d % modulus);
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockCount>(acc, predicate);
            auto const expected_result = static_cast<int>(block_thread_extent1_d / modulus);
            ALPAKA_CHECK(*success, expected_result == result);
        }
        {
            Idx const modulus = 3u;
            auto const predicate = static_cast<int>(block_thread_idx1_d % modulus);
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockCount>(acc, predicate);
            auto const expected_result = static_cast<int>(
                block_thread_extent1_d - ((block_thread_extent1_d + modulus - static_cast<Idx>(1u)) / modulus));
            ALPAKA_CHECK(*success, expected_result == result);
        }

        // syncBlockThreadsPredicate<alpaka::BlockAnd>
        {
            int const predicate = 1;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, predicate);
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate = 0;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, predicate);
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            int const predicate = block_thread_idx1_d != 0;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, predicate);
            ALPAKA_CHECK(*success, result == 0);
        }

        // syncBlockThreadsPredicate<alpaka::BlockOr>
        {
            int const predicate = 1;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, predicate);
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate = 0;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, predicate);
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            auto const predicate = static_cast<int>(block_thread_idx1_d != 1);
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, predicate);
            ALPAKA_CHECK(*success, result == 1);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("synchronizePredicate", "[blockSync]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    BlockSyncPredicateTestKernel kernel;

    // 4^Dim
    {
        alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(4u)));

        REQUIRE(fixture(kernel));
    }

    // 1^Dim
    {
        alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

        REQUIRE(fixture(kernel));
    }
}
