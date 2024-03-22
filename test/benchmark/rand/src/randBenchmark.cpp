/* Copyright 2022 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/rand/Traits.hpp>
#include <alpaka/test/KernelExecutionBenchmarkFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

class RandBenchmarkKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TIdx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, TIdx numPoints) const
    {
        // Get the global linearized thread idx.
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx
            = static_cast<TIdx>(alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0]);

        // Setup generator engine and distribution.
        auto engine = alpaka::rand::engine::createDefault(acc, 42, linearizedGlobalThreadIdx);
        auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));

        float number = 0;
        for(TIdx i = linearizedGlobalThreadIdx; i < numPoints; i += static_cast<TIdx>(globalThreadExtent.prod()))
        {
            number += dist(engine);
        }

        alpaka::atomicAdd(
            acc,
            result,
            number); // TODO: we're measuring the atomicAdd time too, this is not what we want
    }
};

// TODO: This takes an enormous time to finish and is probably useless anyway:
//   TEMPLATE_LIST_TEST_CASE("defaultRandomGeneratorBenchmark", "[randBenchmark]", alpaka::test::TestAccs)
// Running the benchmark on a single default accelerator instead
TEST_CASE("defaultRandomGeneratorBenchmark", "[randBenchmark]")
{
    //    using Acc = TestType;
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, std::size_t>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    Idx const numThreads = std::thread::hardware_concurrency(); // TODO: GPU?
    std::cout << "Hardware threads: " << numThreads << std::endl;

#ifdef ALPAKA_CI // Reduced benchmark set for automated test runs.
    unsigned const numPoints = GENERATE(10u, 1'000'000u);
#else
    unsigned const numPoints = GENERATE(10u, 100000u, 1'000'000u, 10'000'000u, 100'000'000u, 1'000'000'000u);
#endif

    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
        dev,
        Vec::all(numThreads * numThreads),
        Vec::all(numThreads),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    alpaka::test::KernelExecutionBenchmarkFixture<Acc> fixture(workdiv);

    RandBenchmarkKernel kernel;

    float result = 0.0f;

    REQUIRE(fixture(kernel, "Random sequence N=" + std::to_string(numPoints), result, numPoints));
    // TODO: Actually check the result
    std::cout << "\ntemp debug normalized result = " << result / static_cast<float>(numPoints)
              << " should probably converge to 0.5." << std::flush;
}
