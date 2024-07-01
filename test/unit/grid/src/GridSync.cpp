/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/grid/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class GridSyncTestKernel
{
public:
    static constexpr std::uint8_t blockThreadExtentPerDim()
    {
        return 2u;
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T* array) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;

        // Get the index of the current thread within the grid and the grid extent and map them to 1D.
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdx1D = alpaka::mapIdx<1u>(gridThreadIdx, gridThreadExtent)[0u];
        auto const gridThreadExtent1D = gridThreadExtent.prod();


        // Write the thread index into the shared array.
        array[gridThreadIdx1D] = static_cast<T>(gridThreadIdx1D);

        // Synchronize the threads in the block.
        alpaka::syncGridThreads(acc);

        // All other threads within the block should now have written their index into the shared memory.
        for(auto i = static_cast<Idx>(0u); i < gridThreadExtent1D; ++i)
        {
            ALPAKA_CHECK(*success, static_cast<Idx>(array[i]) == i);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("synchronize", "[gridSync]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // Select the first device available on a system, for the chosen accelerator
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = getDevByIdx(platformAcc, 0u);

    // Check if accelerator supports cooperative launch, don't do anything if it doesn't
    if(alpaka::trait::GetAccDevProps<Acc>::getAccDevProps(devAcc).m_cooperativeLaunch)
    {
        auto const blockThreadExtentMax = alpaka::getAccDevProps<Acc>(devAcc).m_blockThreadExtentMax;
        auto threadsPerBlock = alpaka::elementwise_min(
            blockThreadExtentMax,
            alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(GridSyncTestKernel::blockThreadExtentPerDim())));

        auto elementsPerThread = alpaka::Vec<Dim, Idx>::all(1);
        auto blocksPerGrid = alpaka::Vec<Dim, Idx>::all(1);
        blocksPerGrid[0] = 200;

        // Allocate memory on the device.
        alpaka::Vec<alpaka::DimInt<1>, Idx> bufferExtent{
            blocksPerGrid.prod() * threadsPerBlock.prod() * elementsPerThread.prod()};
        auto deviceMemory = alpaka::allocBuf<Idx, Idx>(devAcc, bufferExtent);

        GridSyncTestKernel kernel;

        bool success = false;

        int maxBlocks = alpaka::getMaxActiveBlocks<Acc>(
            devAcc,
            kernel,
            threadsPerBlock,
            elementsPerThread,
            &success,
            alpaka::getPtrNative(deviceMemory));

        blocksPerGrid[0] = std::min(static_cast<Idx>(maxBlocks), blocksPerGrid[0]);
        constexpr bool IsCooperative = true;
        alpaka::test::KernelExecutionFixture<Acc, IsCooperative> fixture(
            alpaka::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread});

        REQUIRE(fixture(kernel, alpaka::getPtrNative(deviceMemory)));
    }
}
