/**
 * \file
 * Copyright 2015-2019 Benjamin Worpitz
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
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>


//#############################################################################
class BlockSharedMemDynTestKernel
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
        // Assure that the pointer is non null.
        auto && a = alpaka::block::shared::dyn::getMem<std::uint32_t>(acc);
        ALPAKA_CHECK(*success, static_cast<std::uint32_t *>(nullptr) != a);

        // Each call should return the same pointer ...
        auto && b = alpaka::block::shared::dyn::getMem<std::uint32_t>(acc);
        ALPAKA_CHECK(*success, a == b);

        // ... even for different types.
        auto && c = alpaka::block::shared::dyn::getMem<float>(acc);
        ALPAKA_CHECK(*success, a == reinterpret_cast<std::uint32_t *>(c));
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
                BlockSharedMemDynTestKernel,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                template<
                    typename TVec>
                ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                    BlockSharedMemDynTestKernel const & blockSharedMemDyn,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    bool * success)
                -> idx::Idx<TAcc>
                {
                    alpaka::ignore_unused(blockSharedMemDyn);
                    alpaka::ignore_unused(success);
                    return
                        static_cast<idx::Idx<TAcc>>(sizeof(std::uint32_t)) * blockThreadExtent.prod() * threadElemExtent.prod();
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
        alpaka::vec::Vec<Dim, Idx>::ones());

    BlockSharedMemDynTestKernel kernel;

    REQUIRE(
        fixture(
            kernel));
}
};

TEST_CASE( "sameNonNullAdress", "[blockSharedMemDyn]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}
