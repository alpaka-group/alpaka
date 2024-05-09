/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#include <alpaka/acc/AccCpuOmp2Threads.hpp>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuTbbBlocks.hpp>
#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/kernel/KernelBundle.hpp>
#include <alpaka/kernel/KernelFunctionAttributes.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

struct TestKernelWithManyRegisters
{
    template<typename TAcc>
    [[maybe_unused]] ALPAKA_FN_ACC auto operator()(TAcc const&, std::size_t sizeOfArray) const -> void
    {
        double var0 = 1.0;
        double var1 = 2.0;
        double var2 = 3.0;

        // Define many variables and use some calculations in order to prevent compiler optimization and make the
        // kernel use many registers (around 80 on sm_52). Using many registers per SM decreases the max number of
        // threads per block while this kernel is being run.

        // TODO: Use function templates to parametrize and shorten the code!
        double var3 = var2 + fmod(var2, 5);
        double var4 = var3 + fmod(var3, 5);
        double var5 = var4 + fmod(var4, 5);
        double var6 = var5 + fmod(var5, 5);
        double var7 = var6 + fmod(var6, 5);
        double var8 = var7 + fmod(var7, 5);
        double var9 = var8 + fmod(var8, 5);
        double var10 = var9 + fmod(var9, 5);
        double var11 = var10 + fmod(var10, 5);
        double var12 = var11 + fmod(var11, 5);
        double var13 = var12 + fmod(var12, 5);
        double var14 = var13 + fmod(var13, 5);
        double var15 = var14 + fmod(var14, 5);
        double var16 = var15 + fmod(var15, 5);
        double var17 = var16 + fmod(var16, 5);
        double var18 = var17 + fmod(var17, 5);
        double var19 = var18 + fmod(var18, 5);
        double var20 = var19 + fmod(var19, 5);
        double var21 = var20 + fmod(var20, 5);
        double var22 = var21 + fmod(var21, 5);
        double var23 = var22 + fmod(var22, 5);
        double var24 = var23 + fmod(var23, 5);
        double var25 = var24 + fmod(var24, 5);
        double var26 = var25 + fmod(var25, 5);
        double var27 = var26 + fmod(var26, 5);
        double var28 = var27 + fmod(var27, 5);
        double var29 = var28 + fmod(var28, 5);
        double var30 = var29 + fmod(var29, 5);
        double var31 = var30 + fmod(var30, 5);
        double var32 = var31 + fmod(var31, 5);
        double var33 = var32 + fmod(var32, 5);
        double var34 = var33 + fmod(var33, 5);
        double var35 = var34 + fmod(var34, 5);

        double sum = var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12
                     + var13 + var14 + var15 + var16 + var17 + var18 + var19 + var20 + var21 + var22 + var23 + var24
                     + var25 + var26 + var27 + var28 + var29 + var30 + var31 + var32 + var33 + var34 + var35;
        printf("The sum is %5.2f, the argument is %lu ", sum, sizeOfArray);
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("getFunctionAttributes.1D.withIdx", "[workDivKernel]", TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    TestKernelWithManyRegisters kernel;
    auto const& bundeledKernel = alpaka::KernelBundle(kernel, 200ul);

    // Test getValidWorkDivForKernel for 1024 threads per grid
    auto const workDiv = alpaka::getValidWorkDivForKernel<Acc>(dev, bundeledKernel, Vec{1024}, Vec{1});
    // Test validity
    auto const isValid = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, workDiv);
    CHECK(isValid == true);

    if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt>)
    {
        // Get threads per block from the workDiv found by examining the kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent[0];
        // Get hard limits
        auto const kernelFunctionAttributes = alpaka::getFunctionAttributes<Acc>(bundeledKernel);
        auto const threadsPerBlockLimit = kernelFunctionAttributes.maxThreadsPerBlock;
        // Compare the workdiv value calculated using the kernel function to the device max limits for threads per
        // block. The calculated value using kernel function is certainly less than or equal to the device max limits.
        // For a common Nvidia GPU, expected values for threadsPerBlock and threadsPerBlockLimit are 512 and 1024.
        // Depending on the GPU or the compiler the test below might fail because threadsPerBlock can be equal to
        // threadsPerBlockLimit, which is the max device limit.
        CHECK(threadsPerBlock < static_cast<Idx>(threadsPerBlockLimit));
    }
    else if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuHipRt>)
    {
        // Get threads per block from the workDiv found by examining the kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent[0];
        // Get hard limits
        auto const kernelFunctionAttributes = alpaka::getFunctionAttributes<Acc>(bundeledKernel);
        auto const threadsPerBlockLimit = kernelFunctionAttributes.maxThreadsPerBlock;
        // Compare the workdiv value calculated using the kernel function to the device max limits for threads per
        // block. The calculated value using kernel function is certainly less than or equal to the device max limits.
        // For a common AMD GPU, expected values for threadsPerBlock and threadsPerBlockLimit are 1024 and 1024.
        // Depending on the GPU or the compiler the test below might fail because threadsPerBlock can be less than
        // threadsPerBlockLimit, which is the max device limit.
        CHECK(threadsPerBlock == static_cast<Idx>(threadsPerBlockLimit));
    }
    else if constexpr(alpaka::accMatchesTags<
                          Acc,
                          alpaka::TagCpuSerial,
                          alpaka::TagCpuThreads,
                          alpaka::TagCpuOmp2Threads,
                          alpaka::TagCpuTbbBlocks>)
    {
        // CPU must have only 1 thread per block. In other words, number of blocks is equal to number of threads.
        CHECK(workDiv == WorkDiv{Vec{1024}, Vec{1}, Vec{1}});
        // Test a new 1D workdiv. Threads per block can not be larger than 1 for CPU. Hence 2 is not valid.
        auto const& workDiv1DUsingInitList = alpaka::WorkDivMembers{Vec{1024}, Vec{2}, Vec{1}};
        auto const isWorkDivValidForCPUKernel
            = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, workDiv1DUsingInitList);
        CHECK(isWorkDivValidForCPUKernel == false);
    }
}

using TestAccs2D = alpaka::test::EnabledAccs<alpaka::DimInt<2u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("getFunctionAttributes.2D.withIdx", "[workDivKernel]", TestAccs2D)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    TestKernelWithManyRegisters kernel;
    size_t val(200ul);
    auto const& bundeledKernel = alpaka::KernelBundle(kernel, val);

    // Test getValidWorkDivForKernel function for 1024x2 threads per grid.
    auto const workDiv = alpaka::getValidWorkDivForKernel<Acc>(dev, bundeledKernel, Vec{1024, 2}, Vec{1, 1});
    // Test isValidWorkDivKernel function
    auto const isValid = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, workDiv);
    CHECK(isValid == true);

    if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt>)
    {
        // Expected valid workdiv for this kernel. These values might change depending on the GPU type and compiler
        // therefore commented out. CHECK(workDiv == WorkDiv{Vec{4, 1}, Vec{256, 2}, Vec{1, 1}});

        // Get calculated threads per block from the workDiv found by examining kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent.prod();
        // Get hard limits
        auto const kernelFunctionAttributes = alpaka::getFunctionAttributes<Acc>(bundeledKernel);
        auto const threadsPerBlockLimit = kernelFunctionAttributes.maxThreadsPerBlock;
        // Depending on the GPU type or the compiler the test below might fail because threadsPerBlock can be equal to
        // threadsPerBlockLimit, which is the max device limit.
        CHECK(threadsPerBlock < static_cast<Idx>(threadsPerBlockLimit));
    }
    else if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuHipRt>)
    {
        // Expected valid workdiv for this kernel. These values might change depending on the GPU type and compiler
        // therefore commented out. CHECK(workDiv == WorkDiv{Vec{2, 1}, Vec{512, 2}, Vec{1, 1}});

        // Get calculated threads per block from the workDiv found by examining the kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent.prod();
        // Get hard limits
        auto const kernelFunctionAttributes = alpaka::getFunctionAttributes<Acc>(bundeledKernel);
        auto const threadsPerBlockLimit = kernelFunctionAttributes.maxThreadsPerBlock;
        // Depending on the GPU type or the compiler this test might fail because threadsPerBlock can be less than
        // threadsPerBlockLimit, which is the max device limit.
        CHECK(threadsPerBlock == static_cast<Idx>(threadsPerBlockLimit));
    }
    else if constexpr(alpaka::accMatchesTags<
                          Acc,
                          alpaka::TagCpuSerial,
                          alpaka::TagCpuThreads,
                          alpaka::TagCpuOmp2Threads,
                          alpaka::TagCpuTbbBlocks>)
    {
        // CPU must have only 1 thread per block. In other words, number of blocks is equal to number of threads.
        CHECK(workDiv == WorkDiv{Vec{1024, 2}, Vec{1, 1}, Vec{1, 1}});
        // Test a new 2D workdiv. Threads per block can not be larger than 1 for CPU. Hence 2x1 threads is not valid.
        auto const& invalidWorkDiv2D = WorkDiv{Vec{2048, 1}, Vec{2, 1}, Vec{1, 1}};
        auto const isWorkDivValidForCpu = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, invalidWorkDiv2D);
        CHECK(isWorkDivValidForCpu == false);
    }
}
