/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/kernel/KernelBundle.hpp"

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuCudaRt.hpp>
#include <alpaka/acc/AccGpuHipRt.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

template<typename T, uint64_t size>
struct cheapArray
{
    T data[size];

    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) -> T&
    {
        return data[index];
    }

    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per constant reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) const -> T const&
    {
        return data[index];
    }
};

struct TestKernel
{
    template<typename TAcc>
    [[maybe_unused]] ALPAKA_FN_ACC auto operator()(TAcc const& acc, std::size_t sizeOfArray) const -> void
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;
        using Vec2 = alpaka::Vec<alpaka::DimInt<2u>, Idx>;
        using Vec3 = alpaka::Vec<alpaka::DimInt<3u>, Idx>;
        // auto const platformAcc = alpaka::Platform<TAcc>{};
        // auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
        std::size_t dummyVar = sizeOfArray;
        std::size_t dummyArr[1024] = {0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8};
        auto& sdata(alpaka::declareSharedVar<cheapArray<double, 5000>, __COUNTER__>(acc));
        // auto& sdata2(alpaka::declareSharedVar<cheapArray<double, 5000>, __COUNTER__>(acc));
        std::size_t dummySum = 3u;
        for(std::size_t i = 0; i < dummyVar; i++)
        {
            std::size_t partialSum = dummySum;
            sdata[i] = double(i);
            dummySum += i + partialSum + dummyArr[dummyVar % 3];
        }


        std::size_t dummyArrX[1000] = {0, 1, 2, 3, 4};
        std::size_t dummyArrY[5000] = {0, 1, 2, 3, 4};


        std::size_t dummySum2 = 4u;
        // for(std::size_t i = 0; i < dummyVar2; i++)
        // {
        //     std::size_t partialSum2 = dummySum2;
        //     dummySum2 += i + partialSum2 + dummyArr2[dummyVar2 % 5] + dummyArr5[dummyVar5 % 5] + sdata[i];
        // }
        auto v2 = Vec2{2u, 4u};
        auto v3 = Vec3{2u, 4u, 6u};

        std::size_t dummyVar3 = 40u;
        std::size_t dummyArr3[5000] = {0, 1, 2, 3, 4};
        std::size_t dummyVar4 = 40u + dummySum2;
        std::size_t dummySum3 = 4u;
        for(std::size_t i = 0; i < dummyVar3; i++)
        {
            std::size_t partialSum3 = dummySum3 ^ 2;
            dummySum3 += i + partialSum3 + dummyArr3[dummyVar3 % 5] + dummySum2 + dummyArrX[dummyVar3 % 5]
                         + dummyArrY[dummyVar3 % 5];
        }
        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec1 const globalThreadExtent = Vec1{256};
        // Vec const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const globalBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
        // auto const globalThreadExtent2 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalBlockExtent3 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 1;
        auto const globalThreadExtent3 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 2;
        auto const globalBlockExtent4 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 3;
        auto const globalThreadExtent4 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 4;
        auto const globalBlockExtent5 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 5;
        auto const globalThreadExtent5 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        Vec1 const globalThreadExtent22 = Vec1{256} * 2;
        Vec1 const globalThreadExtent23 = Vec1{256} * 3;
        Vec1 const globalThreadExtent24 = Vec1{256} * 4;
        Vec1 const globalThreadExtent25 = Vec1{256} * 5;
        Vec1 const globalThreadExtent26 = Vec1{256} * 6;
        Vec1 const globalThreadExtent27 = Vec1{256} * 7;
        Vec1 const globalThreadExtent28 = Vec1{256} * 8;
        Vec1 const globalThreadExtent29 = Vec1{256} * 9;
        Vec1 const globalThreadExtent30 = Vec1{256} * 10;
        Vec1 const globalThreadExtent31 = Vec1{256} * 11;
        // Map the three dimensional thread index into a
        // one dimensional thread index space. We call it
        // linearize the thread index.
        Vec1 const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);
        std::size_t dummyVar6 = 45u;
        for(std::size_t i = 0; i < dummyVar3; i++)
        {
            std::size_t partialSum4 = std::size_t(sdata[21]) + dummySum3 ^ 2 + dummyVar6;
            dummySum3 += i + partialSum4 + dummyArr3[dummyVar3 % 5] + dummySum2 + dummyArrX[dummyVar3 % 5]
                         + dummyArrY[dummyVar3 % 5] + dummyVar6;
        }

        auto const globalBlockExtent11 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
        auto const globalThreadExtent12 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalBlockExtent13 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 1;
        auto const globalThreadExtent13 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 2;
        auto const globalBlockExtent14 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 3;
        auto const globalThreadExtent14 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 4;
        // auto const globalBlockExtent15 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 5;
        // auto const globalThreadExtent15 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent41 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 6;
        auto const globalThreadExtent41 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent42 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 1;
        auto const globalThreadExtent42 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 2;
        auto const globalBlockExtent43 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 3;
        auto const globalThreadExtent43 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 4;
        auto const globalBlockExtent44 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 5;
        auto const globalThreadExtent44 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent45 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 6;
        auto const globalThreadExtent45 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent46 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 1;
        auto const globalThreadExtent46 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 2;
        auto const globalBlockExtent47 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 3;
        auto const globalThreadExtent47 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 4;
        auto const globalBlockExtent48 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 5;
        auto const globalThreadExtent48 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent49 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 6;
        auto const globalThreadExtent49 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent50 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 1;
        auto const globalThreadExtent50 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 2;
        auto const globalBlockExtent51 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 3;
        auto const globalThreadExtent51 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 4;
        auto const globalBlockExtent52 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 5;
        auto const globalThreadExtent52 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent411 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 9;
        auto const globalThreadExtent411 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        auto const globalBlockExtent421 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 9;
        auto const globalThreadExtent421 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 5;
        auto const globalBlockExtent431 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 9;
        auto const globalThreadExtent431 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 4;
        auto const globalBlockExtent441 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 9;
        auto const globalThreadExtent441 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 5;
        auto const globalBlockExtent451 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 9;
        auto const globalThreadExtent451 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 2;
        auto const globalBlockExtent461 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 9;
        auto const globalThreadExtent461 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 6;
        auto const globalBlockExtent471 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 4;
        auto const globalThreadExtent471 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 8;
        auto const globalBlockExtent481 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 4;
        auto const globalThreadExtent481 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 8;
        auto const globalBlockExtent491 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 4;
        auto const globalThreadExtent491 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 7;
        auto const globalBlockExtent501 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] + 4;
        auto const globalThreadExtent501 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0] + 6;
        auto const globalBlockExtent511 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] + 3;
        auto const globalThreadExtent511 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + 8;
        auto const globalBlockExtent521 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] + 9;
        auto const globalThreadExtent521 = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[2] + 6;
        // Each thread prints a hello world to the terminal
        // together with the global index of the thread in
        // each dimension and the linearized global index.
        // Mind, that alpaka uses the mathematical index
        // order [z][y][x] where the last index is the fast one.
        printf(
            "[z:%u, y:%u, x:%u][linear:%u] [dummy var:%u] Global Block Extent: %u GlobalThreadExtent: %u %u %u %u %u "
            "%u %u %u %u %u %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f  "
            "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f  "
            " Test Kernel \n",
            static_cast<unsigned>(globalThreadIdx[0u]),
            static_cast<unsigned>(globalThreadIdx[1u]),
            static_cast<unsigned>(globalThreadIdx[2u]),
            static_cast<unsigned>(linearizedGlobalThreadIdx[0u]),
            static_cast<unsigned>(dummyVar4 + dummySum + dummySum2 + dummySum3 ^ 2 + dummyArrY[33]),
            static_cast<unsigned>(globalBlockExtent[0]),
            static_cast<unsigned>(globalThreadExtent22[0u]),
            static_cast<unsigned>(globalThreadExtent22[0u]),
            static_cast<unsigned>(globalThreadExtent22[0u]),
            static_cast<unsigned>(globalThreadExtent23[0u]),
            static_cast<unsigned>(globalThreadExtent24[0u]),
            static_cast<unsigned>(globalThreadExtent25[0u]),
            static_cast<unsigned>(globalThreadExtent26[0u]),
            static_cast<unsigned>(globalThreadExtent27[0u]),
            static_cast<unsigned>(globalThreadExtent28[0u]),
            static_cast<unsigned>(globalThreadExtent29[0u]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent30[0u]) + static_cast<double>(v3[0] ^ v2[2]) * sdata[24]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent31[0u]) + static_cast<double>(v2[0] ^ v2[0]) * sdata[16]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent3) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[15]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent3) * sdata[19] * static_cast<double>(v2[0] ^ v2[0]) * sdata[24]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent11[0]) * sdata[18] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[15]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent11[0]) * sdata[18] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[15]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent12[0]) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[24]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent13) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[15]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent13) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[24]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent14) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[25]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent14) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[26]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent41) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[27]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent41) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[28]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent42) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[29]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent42) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[30]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent43) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[31]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent43) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[24]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent44) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[32]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent44) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[33]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent45) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[34]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent45) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[35]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent46) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[36]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent46) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[37]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent47) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[38]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent47) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[39]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent48) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[40]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent48) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[41]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent49) * sdata[10] * static_cast<double>(v2[0] ^ v2[0]) * sdata[42]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent49) * sdata[10] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[43]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent50) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[44]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent50) * sdata[11] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[45]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent51) * sdata[12] * static_cast<double>(v2[0] ^ v2[0]) * sdata[46]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent51) * sdata[13] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[47]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent52) * sdata[14] * static_cast<double>(v2[0] ^ v2[0]) * sdata[48]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent52) * sdata[15] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[49]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent411) * sdata[16] * static_cast<double>(v2[0] ^ v2[0]) * sdata[0]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent411) * sdata[17] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[1]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent421) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[2]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent421) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[3]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent431) * sdata[21] * static_cast<double>(v2[0] ^ v2[0]) * sdata[4]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent431) * sdata[49] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[5]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent441) * sdata[48] * static_cast<double>(v2[0] ^ v2[0]) * sdata[5]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent441) * sdata[49] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[6]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent451) * sdata[38] * static_cast<double>(v2[0] ^ v2[0]) * sdata[7]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent451) * sdata[39] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[8]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent461) * sdata[38] * static_cast<double>(v2[0] ^ v2[0]) * sdata[9]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent461) * sdata[39] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[12]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent471) * sdata[38] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[13]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent471) * sdata[29] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[14]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent481) * sdata[28] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[15]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent481) * sdata[29] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[16]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent491) * sdata[12] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[34]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent491) * sdata[13] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[34]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent501) * sdata[14] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[35]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent501) * sdata[15] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[35]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent511) * sdata[16] * static_cast<double>(v2[0] ^ v2[0]) * sdata[6]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent511) * sdata[17] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[7]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent521) * sdata[18] * static_cast<double>(v2[0] ^ v2[0]) * sdata[8]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent521) * sdata[19] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[9]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent4) * sdata[22] * static_cast<double>(v2[0] ^ v2[0]) * sdata[12]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent4) * sdata[23] * static_cast<double>(v2[0] ^ v2[0]) * sdata[13]),
            static_cast<double>(
                static_cast<double>(globalBlockExtent5) * sdata[20] * static_cast<double>(v2[0] ^ v2[0]) * sdata[14]),
            static_cast<double>(
                static_cast<double>(globalThreadExtent5) * sdata[21] * static_cast<double>(v2[0] ^ v2[0])
                * sdata[14]));
    }
};

TEMPLATE_LIST_TEST_CASE("getMaxThreadsInBlockForKernel.1D.withIdx", "[workDivKernel]", alpaka::test::TestAccs)
{
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    TestKernel kernel;
    printf("gWorkDivForKernel test1\n");

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    printf("gWorkDivForKernel CUDA ENABLE defined\n");
    using TApi = alpaka::ApiCudaRt;
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    printf("gtWorkDivForKernel HIP ENABLE defined\n");
    using TApi = alpaka::ApiHipRt;
#        endif

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    size_t kernelArg = 200ul;
    printf("gWorkDivForKernel test3\n");
    using Vec = alpaka::Vec<Dim, Idx>;
    // alpaka::getMaxThreadsInBlockForKernel<TApi, Acc, TestKernel, std::size_t>(kernel, kernelArg);
    auto const bundeledKernel = alpaka::makeKernelBundle<Acc>(kernel, kernelArg);
    alpaka::getValidWorkDivForKernel(bundeledKernel);
#    endif
}
#endif
