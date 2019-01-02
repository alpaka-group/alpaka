/**
 * \file
 * Copyright 2017-2019 Benjamin Worpitz
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
class BlockSyncTestKernel
{
public:
    static const std::uint8_t gridThreadExtentPerDim = 4u;

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
        auto const blockThreadIdx = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadIdx1D = alpaka::idx::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockThreadExtent1D = blockThreadExtent.prod();

        // Allocate shared memory.
        Idx * const pBlockSharedArray = alpaka::block::shared::dyn::getMem<Idx>(acc);
   
        // Write the thread index into the shared memory.
        pBlockSharedArray[blockThreadIdx1D] = blockThreadIdx1D;

        // Synchronize the threads in the block.
        alpaka::block::sync::syncBlockThreads(acc);

        // All other threads within the block should now have written their index into the shared memory.
        for(auto i(static_cast<Idx>(0u)); i < blockThreadExtent1D; ++i)
        {
            ALPAKA_CHECK(*success, pBlockSharedArray[i] == i);
        }
    }
};

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared dynamic memory for a kernel.
            template<
                typename TAcc>
            struct BlockSharedMemDynSizeBytes<
                BlockSyncTestKernel,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                template<
                    typename TVec>
                ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                    BlockSyncTestKernel const & blockSharedMemDyn,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    bool * success)
                -> idx::Idx<TAcc>
                {
                    using Idx = alpaka::idx::Idx<TAcc>;

                    alpaka::ignore_unused(blockSharedMemDyn);
                    alpaka::ignore_unused(threadElemExtent);
                    alpaka::ignore_unused(success);
                    return
                        static_cast<idx::Idx<TAcc>>(sizeof(Idx)) * blockThreadExtent.prod();
                }
            };
        }
    }
}

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(BlockSyncTestKernel::gridThreadExtentPerDim)));

    BlockSyncTestKernel kernel;

    REQUIRE(
        fixture(
            kernel));
}
};

TEST_CASE( "synchronize", "[blockSync]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}
