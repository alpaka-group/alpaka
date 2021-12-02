/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
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

class BlockSyncTestKernel
{
public:
    static const std::uint8_t grid_thread_extent_per_dim = 4u;

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

        // Allocate shared memory.
        auto* const p_block_shared_array = alpaka::getDynSharedMem<Idx>(acc);

        // Write the thread index into the shared memory.
        p_block_shared_array[block_thread_idx1_d] = block_thread_idx1_d;

        // Synchronize the threads in the block.
        alpaka::syncBlockThreads(acc);

        // All other threads within the block should now have written their index into the shared memory.
        for(auto i = static_cast<Idx>(0u); i < block_thread_extent1_d; ++i)
        {
            ALPAKA_CHECK(*success, p_block_shared_array[i] == i);
        }
    }
};

namespace alpaka
{
    namespace traits
    {
        //! The trait for getting the size of the block shared dynamic memory for a kernel.
        template<typename TAcc>
        struct BlockSharedMemDynSizeBytes<BlockSyncTestKernel, TAcc>
        {
            //! \return The size of the shared memory allocated for a block.
            template<typename TVec>
            ALPAKA_FN_HOST_ACC static auto get_block_shared_mem_dyn_size_bytes(
                BlockSyncTestKernel const& block_shared_mem_dyn,
                TVec const& block_thread_extent,
                TVec const& thread_elem_extent,
                bool* success) -> std::size_t
            {
                using Idx = alpaka::Idx<TAcc>;

                alpaka::ignore_unused(block_shared_mem_dyn);
                alpaka::ignore_unused(thread_elem_extent);
                alpaka::ignore_unused(success);
                return static_cast<std::size_t>(block_thread_extent.prod()) * sizeof(Idx);
            }
        };
    } // namespace traits
} // namespace alpaka

TEMPLATE_LIST_TEST_CASE("synchronize", "[blockSync]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(BlockSyncTestKernel::grid_thread_extent_per_dim)));

    BlockSyncTestKernel kernel;

    REQUIRE(fixture(kernel));
}
