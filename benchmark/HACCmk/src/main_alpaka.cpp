/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <bit>
#include <chrono>
#include <iostream>

namespace haccmkAlpaka
{
    template<typename T>
    struct DeltaMove
    {
        T x;
        T y;
        T z;

        constexpr DeltaMove operator+(DeltaMove const& other)
        {
            return DeltaMove{x + other.x, y + other.y, z + other.z};
        }
    };

    constexpr void alpakaSimdizedInvoke(auto&& f, alpaka::concepts::SpecializationOf<DeltaMove> auto&&... args)
    {
        alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).x...);
        alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).y...);
        alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).z...);
    }

    template<uint32_t T_width, typename T>
    constexpr auto makeSimdized(DeltaMove<T> const& value)
    {
        using SimdMemberType = ALPAKA_TYPEOF(alpaka::makeSimdized<T_width>(std::declval<T>()));
        DeltaMove<SimdMemberType> result;
        alpakaSimdizedInvoke([](alpaka::concepts::Simd auto& lhs, auto const& rhs) { lhs = rhs; }, result, value);
        return result;
    }

    struct Step10Kernel
    {
        ALPAKA_FN_ACC auto operator()(
            auto const& acc,
            int n1,
            int n2,
            alpaka::concepts::IMdSpan auto xx,
            alpaka::concepts::IMdSpan auto yy,
            alpaka::concepts::IMdSpan auto zz,
            alpaka::concepts::IMdSpan auto mass,
            alpaka::concepts::IMdSpan auto vx2,
            alpaka::concepts::IMdSpan auto vy2,
            alpaka::concepts::IMdSpan auto vz2,
            float fsrrmax2,
            float mp_rsm2,
            float fcoeff) const
        {
            using namespace alpaka;

            constexpr float const ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313,
                                  ma4 = 0.0000605491, ma5 = -0.00000147177;

            for(auto i :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearThreadsInGrid, alpaka::IdxRange{n1}))
            {
                auto move = DeltaMove<float>{0, 0, 0};

                float xxi = xx[i];
                float yyi = yy[i];
                float zzi = zz[i];

                auto simdGrid = onAcc::SimdAlgo{onAcc::worker::allThreads};
                move = simdGrid.transformReduce(
                    acc,
                    Vec{n2},
                    move,
                    std::plus{},
                    [&](auto const&,
                        concepts::Simd auto const& simd_xx1,
                        concepts::Simd auto const& simd_yy1,
                        concepts::Simd auto const& simd_zz1,
                        concepts::Simd auto const& simd_mass1) constexpr
                    {
                        concepts::Simd auto dxc = simd_xx1 - xxi;
                        concepts::Simd auto dyc = simd_yy1 - yyi;
                        concepts::Simd auto dzc = simd_zz1 - zzi;

                        concepts::Simd auto r2 = dxc * dxc + dyc * dyc + dzc * dzc;

                        using SimdType = ALPAKA_TYPEOF(simd_mass1);
                        concepts::Simd auto m = SimdType::fill(0.f);

                        where(r2 < fsrrmax2, m) = simd_mass1;

                        alpaka::concepts::Simd auto tmp = r2 + mp_rsm2;
                        concepts::Simd auto bar = alpaka::math::sqrt(tmp);

                        concepts::Simd auto p = float{1.0} / (tmp * bar);

                        concepts::Simd auto f
                            = p - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5)))));
                        concepts::Simd auto fac = SimdType::fill(0.f);
                        where(r2 > 0.0f, fac) = m * f;

                        using SimdizedType = decltype(makeSimdized<SimdType::width()>(move));
                        return SimdizedType{(fac * dxc), (fac * dyc), (fac * dzc)};
                    },
                    xx,
                    yy,
                    zz,
                    mass);

                /* No `+=` is used to be comparable with the original version.
                 * This differs to the original publication but is necessary due to an OpenMP bug in the original
                 * implementation.
                 */
                vx2[i] = move.x * fcoeff;
                vy2[i] = move.y * fcoeff;
                vz2[i] = move.z * fcoeff;
            }
        }
    };

    void haccmk(
        auto devAcc,
        auto exec,
        int const repeat,
        int const n1,
        int const n2,
        auto& xx,
        auto& yy,
        auto& zz,
        auto& mass,
        auto& vx2,
        auto& vy2,
        auto& vz2,
        float const fsrmax,
        float const mp_rsm,
        float const fcoeff)
    {
        using namespace alpaka;

        // a blocking queue is used to reduce the starting overhead for CPU devices
        onHost::Queue queue = devAcc.makeQueue(queueKind::blocking);

        auto d_xx = onHost::alloc<float>(devAcc, n2);
        auto d_yy = onHost::alloc<float>(devAcc, n2);
        auto d_zz = onHost::alloc<float>(devAcc, n2);
        auto d_mass = onHost::alloc<float>(devAcc, n2);
        auto d_vx2 = onHost::alloc<float>(devAcc, n1);
        auto d_vy2 = onHost::alloc<float>(devAcc, n1);
        auto d_vz2 = onHost::alloc<float>(devAcc, n1);

        onHost::memcpy(queue, d_xx, xx);
        onHost::memcpy(queue, d_yy, yy);
        onHost::memcpy(queue, d_zz, zz);
        onHost::memcpy(queue, d_mass, mass);

        auto devProps = devAcc.getDeviceProperties();
        // 8 is used for GPUs to provide at least frames to each multiprocessor
        uint32_t suggestedNumFrames = devProps.multiProcessorCount * 8u;
        // crop to the outer loop count if needed
        int numeFrames = std::min(n1, static_cast<int>(suggestedNumFrames));
        uint32_t possibleChunkSize = std::bit_floor(static_cast<uint32_t>(alpaka::divExZero(n1, numeFrames)));
        uint32_t numElementsPerThread = getNumElemPerThread<float>(devAcc.getApi(), devAcc.getDeviceKind());
        uint32_t frameExtent = alpaka::divExZero(possibleChunkSize, numElementsPerThread);
        frameExtent = std::bit_floor(frameExtent);
        frameExtent = frameExtent < devProps.warpSize ? devProps.warpSize : frameExtent;
        auto frameSpec = onHost::FrameSpec{numeFrames, static_cast<int>(frameExtent), exec};

        std::cout << "FrameSpec " << frameSpec << std::endl;

        float total_time = 0.f;

        for(int i = 0; i < repeat; i++)
        {
            // reset output
            onHost::memcpy(queue, d_vx2, vx2, n1);
            onHost::memcpy(queue, d_vy2, vy2, n1);
            onHost::memcpy(queue, d_vz2, vz2, n1);

            onHost::wait(devAcc);
            auto start = std::chrono::steady_clock::now();

            auto const haccKernel = KernelBundle{
                Step10Kernel{},
                n1,
                n2,
                d_xx,
                d_yy,
                d_zz,
                d_mass,
                d_vx2,
                d_vy2,
                d_vz2,
                fsrmax,
                mp_rsm,
                fcoeff};

            queue.enqueue(frameSpec, haccKernel);
            onHost::wait(devAcc);
            auto end = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            // first round is a warmup round
            if(i != 0)
                total_time += time;
        }

        printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / (repeat - 1));
        onHost::memcpy(queue, vx2, d_vx2, n1);
        onHost::memcpy(queue, vy2, d_vy2, n1);
        onHost::memcpy(queue, vz2, d_vz2, n1);
        onHost::wait(devAcc);
    }

    auto example(auto const deviceSpec, auto const exec, int n2, int n1, int repeat) -> int
    {
        using namespace alpaka;

        using IdxVec = Vec<int, 1u>;

        // Define problem size
        IdxVec const extent(n2);

        std::cout << "Outer loop count is set " << n1 << std::endl;
        std::cout << "Inner loop count is set " << n2 << std::endl;
        std::cout << "Element type float" << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device devAcc = devSelector.makeDevice(0);


        auto xx = onHost::allocHost<float>(extent);
        auto yy = onHost::allocHost<float>(extent);
        auto zz = onHost::allocHost<float>(extent);
        auto mass = onHost::allocHost<float>(extent);
        auto vx2 = onHost::allocHost<float>(extent);
        auto vy2 = onHost::allocHost<float>(extent);
        auto vz2 = onHost::allocHost<float>(extent);
        auto vx2_hw = onHost::allocHost<float>(extent);
        auto vy2_hw = onHost::allocHost<float>(extent);
        auto vz2_hw = onHost::allocHost<float>(extent);

        float fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1, dx2, dy2, dz2;
        int i = 0;


        /* Initial data preparation */
        fcoeff = 0.23f;
        fsrrmax2 = 0.5f;
        mp_rsm2 = 0.03f;
        dx1 = 1.0f / static_cast<float>(n2);
        dy1 = 2.0f / static_cast<float>(n2);
        dz1 = 3.0f / static_cast<float>(n2);
        xx[0] = 0.f;
        yy[0] = 0.f;
        zz[0] = 0.f;
        mass[0] = 2.f;

        for(i = 1; i < n2; i++)
        {
            xx[i] = xx[i - 1] + dx1;
            yy[i] = yy[i - 1] + dy1;
            zz[i] = zz[i - 1] + dz1;
            mass[i] = static_cast<float>(i) * 0.01f + xx[i];
        }

        for(i = 0; i < n2; i++)
        {
            vx2[i] = 0.f;
            vy2[i] = 0.f;
            vz2[i] = 0.f;
            vx2_hw[i] = 0.f;
            vy2_hw[i] = 0.f;
            vz2_hw[i] = 0.f;
        }

        for(i = 0; i < n1; ++i)
        {
            hacc::haccmk_gold(n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx2, &dy2, &dz2);
            vx2[i] = vx2[i] + dx2 * fcoeff;
            vy2[i] = vy2[i] + dy2 * fcoeff;
            vz2[i] = vz2[i] + dz2 * fcoeff;
        }

        haccmk(devAcc, exec, repeat, n1, n2, xx, yy, zz, mass, vx2_hw, vy2_hw, vz2_hw, fsrrmax2, mp_rsm2, fcoeff);

        return hacc::verify(n2, vx2, vy2, vz2, vx2_hw, vy2_hw, vz2_hw);
    }
} // namespace haccmkAlpaka

auto main(int argc, char* argv[]) -> int
{
    // naming is taken from the original code
    int n2 = 1;
    int n1 = 1;
    int repeat = 1;

    if(int const ret = hacc::parseCmd(argc, argv, n2, n1, repeat))
        return ret;

    using namespace alpaka;

    /* Execute the example once for each backend (device specification + executor)
     *
     * If you would like to execute it for a single accelerator only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  auto executor = exec::gpuCuda;
     *  return example(deviceSpec, executor, n2, n1, repeat);
     *  @endcode
     *
     * Some examples for device specifications (depending on the active dependencies).
     *
     *   onHost::DeviceSpec{api::host, deviceKind::cpu}
     *   onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu}
     *   onHost::DeviceSpec{api::hip, deviceKind::amdGpu}
     *   onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu}
     *
     * A list of api's and device kinds can be found
     * https://alpaka.readthedocs.io/en/latest/basic/cheatsheet.html#available-apis
     * A list of executors can be found
     * https://alpaka.readthedocs.io/en/latest/basic/cheatsheet.html#executors
     */
    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return haccmkAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), n2, n1, repeat); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
