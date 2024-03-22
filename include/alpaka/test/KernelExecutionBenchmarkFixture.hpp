/* Copyright 2022 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#    error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#    error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/test/Check.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>

#include <string>
#include <utility>

namespace alpaka::test
{
    //! The fixture for executing a kernel on a given accelerator.
    template<typename TAcc>
    class KernelExecutionBenchmarkFixture
    {
    public:
        using Acc = TAcc;
        using Dim = alpaka::Dim<Acc>;
        using Idx = alpaka::Idx<Acc>;
        using Device = Dev<Acc>;
        using Platform = alpaka::Platform<Acc>;
        using Queue = test::DefaultQueue<Device>;
        using WorkDiv = WorkDivMembers<Dim, Idx>;

        KernelExecutionBenchmarkFixture(WorkDiv workDiv) : m_workDiv(std::move(workDiv))
        {
        }

        template<typename TExtent>
        KernelExecutionBenchmarkFixture(TExtent const& extent)
            : KernelExecutionBenchmarkFixture(getValidWorkDiv<Acc>(
                getDevByIdx<Acc>(0u),
                extent,
                Vec<Dim, Idx>::ones(),
                false,
                GridBlockExtentSubDivRestrictions::Unrestricted))
        {
        }

        template<typename TKernelFnObj, typename... TArgs>
        auto operator()(
            TKernelFnObj const& kernelFnObj,
            std::string const& benchmarkName,
            float& result,
            TArgs&&... args) -> bool
        {
            // Allocate result buffers
            auto bufAccResult = allocBuf<float, Idx>(m_device, static_cast<Idx>(1u));
            auto bufHostResult = allocBuf<float, Idx>(m_devHost, static_cast<Idx>(1u));

            int numRuns = 0;
            result = 0.0f;

            // The following block is executed unknown times during estimation phase, then once per benchmark sample
            BENCHMARK_ADVANCED(std::string(benchmarkName))(Catch::Benchmark::Chronometer meter)
            {
                numRuns++;
                memset(m_queue, bufAccResult, 0);
                wait(m_queue);

                // Only the following part is measured as the benchmark part
                meter.measure(
                    [&]
                    {
                        exec<Acc>(
                            m_queue,
                            m_workDiv,
                            kernelFnObj,
                            getPtrNative(bufAccResult),
                            std::forward<TArgs>(args)...); // run the measured kernel
                        wait(m_queue); // wait for the kernel to actually run
                    });

                // Copy the result value to the host
                memcpy(m_queue, bufHostResult, bufAccResult);
                wait(m_queue);

                auto const resultLocal = *getPtrNative(bufHostResult);
                result += resultLocal;
                return resultLocal; // make sure the benchmark call is not optimized away
            };
            result /= static_cast<float>(numRuns);

            return true;
            // TODO: Can we return the result here and read it from Catch2's REQUIRE or something similar? Or are the
            // returns limited to bools?
            //            return result;
        }

    protected:
        PlatformCpu m_platformHost{};
        DevCpu m_devHost{getDevByIdx(m_platformHost, 0)};
        Platform m_platform{};
        Device m_device{getDevByIdx(m_platform, 0)};
        Queue m_queue{m_device};
        WorkDiv m_workDiv;
    };
} // namespace alpaka::test
