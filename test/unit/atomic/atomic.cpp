/* Copyright 2025 Axel Hübl, Benjamin Worpitz, Matthias Werner, Sergei Bastrakov, René Widera, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "atomicHelpers.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

using namespace alpaka::test::unit::atomic;

template<typename T_T1, typename T_T2>
constexpr auto equals(T_T1 a, T_T2 b) -> bool
{
    return a == b;
}

constexpr auto equals(float a, float b) -> bool
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

constexpr auto equals(double a, double b) -> bool
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

template<typename T_AtomicScope, typename T_Op, typename T_Acc, typename T>
constexpr auto testAtomicCall(T_Acc const& acc, bool* success, T& operand, T operandOrig, T value) -> void
{
    auto op = typename T_Op::Op{};

    // check if the function `alpaka::atomicOp<*>` is callable
    {
        // left operand is half of the right
        operand = operandOrig;
        T reference = operand;
        op(&reference, value);

        T const ret = alpaka::onAcc::atomicOp<typename T_Op::Op>(acc, &operand, value, T_AtomicScope{});
        // check that always the old value is returned
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        // check that result in memory is correct
        ALPAKA_CHECK(*success, equals(operand, reference));
    }

    // check if the function `alpaka::atomic*()` is callable
    {
        // left operand is half of the right
        operand = operandOrig;
        T reference = operand;
        op(&reference, value);

        T const ret = T_Op::atomic(acc, &operand, value, T_AtomicScope{});
        // check that always the old value is returned
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        // check that result in memory is correct
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

template<typename T_AtomicScope, typename T_Op, typename T_Acc, typename T>
constexpr auto testAtomicCombinations(T_Acc const& acc, bool* success, T& operand, T operandOrig) -> void
{
    // helper variables to avoid compiler conversion warnings/errors
    constexpr T one = static_cast<T>(1);
    constexpr T two = static_cast<T>(2);
    {
        // left operand is half of the right
        T const value = static_cast<T>(operandOrig / two);
        testAtomicCall<T_AtomicScope, T_Op>(acc, success, operand, operandOrig, value);
    }
    {
        // left operand is twice as large as the right
        T const value = static_cast<T>(operandOrig * two);
        testAtomicCall<T_AtomicScope, T_Op>(acc, success, operand, operandOrig, value);
    }
    {
        // left operand is larger by one
        T const value = static_cast<T>(operandOrig + one);
        testAtomicCall<T_AtomicScope, T_Op>(acc, success, operand, operandOrig, value);
    }
    {
        // left operand is smaller by one
        T const value = static_cast<T>(operandOrig - one);
        testAtomicCall<T_AtomicScope, T_Op>(acc, success, operand, operandOrig, value);
    }
    {
        // both operands are equal
        T const value = operandOrig;
        testAtomicCall<T_AtomicScope, T_Op>(acc, success, operand, operandOrig, value);
    }
}

template<typename T_AtomicScope, typename T_Acc, typename T>
constexpr auto testAtomicCas(T_Acc const& acc, bool* success, T& operand, T operandOrig) -> void
{
    T const value = static_cast<T>(4);

    // with match
    {
        T const compare = operandOrig;
        T const reference = value;
        {
            operand = operandOrig;
            T const ret
                = alpaka::onAcc::atomicOp<alpaka::operation::Cas>(acc, &operand, compare, value, T_AtomicScope{});
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operandOrig;
            T const ret = alpaka::onAcc::atomicCas(acc, &operand, compare, value, T_AtomicScope{});
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
    }

    // without match
    {
        T const compare = static_cast<T>(operandOrig + static_cast<T>(1));
        T const reference = operandOrig;
        {
            operand = operandOrig;
            T const ret
                = alpaka::onAcc::atomicOp<alpaka::operation::Cas>(acc, &operand, compare, value, T_AtomicScope{});
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operandOrig;
            T const ret = alpaka::onAcc::atomicCas(acc, &operand, compare, value, T_AtomicScope{});
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
    }
}

template<typename T_AtomicScope, typename T>
class AtomicTestKernel
{
public:
    ALPAKA_FN_ACC auto operator()(auto const& acc, bool* success, T operandOrig) const -> void
    {
        auto& operand = declareSharedVar<T, uniqueId()>(acc);

        testAtomicCombinations<T_AtomicScope, Add>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Sub>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Exch>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Min>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Max>(acc, success, operand, operandOrig);

        testAtomicCombinations<T_AtomicScope, And>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Or>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Xor>(acc, success, operand, operandOrig);

        if constexpr(std::is_unsigned_v<T>)
        {
            // atomicInc / atomicDec are implemented only for unsigned integer types
            testAtomicCombinations<T_AtomicScope, Inc>(acc, success, operand, operandOrig);
            testAtomicCombinations<T_AtomicScope, Dec>(acc, success, operand, operandOrig);
        }

        testAtomicCas<T_AtomicScope>(acc, success, operand, operandOrig);
    }
};

template<typename T_AtomicScope, std::floating_point T>
class AtomicTestKernel<T_AtomicScope, T>
{
public:
    ALPAKA_FN_ACC auto operator()(auto const& acc, bool* success, T operandOrig) const -> void
    {
        auto& operand = declareSharedVar<T, uniqueId()>(acc);

        testAtomicCombinations<T_AtomicScope, Add>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Sub>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Exch>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Min>(acc, success, operand, operandOrig);
        testAtomicCombinations<T_AtomicScope, Max>(acc, success, operand, operandOrig);

        // Inc, Dec, Or, And, Xor are not supported on float/double types

        testAtomicCas<T_AtomicScope>(acc, success, operand, operandOrig);
    }
};

template<typename T>
struct TestAtomicOperations
{
    static auto testAtomicOperations(auto device, alpaka::concepts::Executor auto exec) -> void
    {
#if ALPAKA_LANG_ONEAPI
        // support for double precision is not guaranteed for sycl devices such as Intel GPUs
        if constexpr(
            std::is_same_v<GetValueType_t<T>, double> && ALPAKA_TYPEOF(alpaka::getApi(device)){} == api::oneApi)
        {
            if(device.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
            {
                SUCCEED(
                    onHost::getName(device)
                    << " does not support double precision.\n Skip benchmark.\n"
                       "For Intel Arc GPUs, use the environment variables `IGC_EnableDPEmulation=1 "
                       "OverrideDefaultFP64Settings=1` to emulate double precision support.\n");
                return;
            }
        }
#endif

        auto queue = device.makeQueue(queueKind::blocking);

        auto status = onHost::allocUnified<bool>(device, 1u);

        status[0] = true;

        T value = static_cast<T>(32);

        queue.enqueue(
            onHost::FrameSpec{1u, 1u, exec},
            KernelBundle{AtomicTestKernel<alpaka::onAcc::scope::Block, T>{}, status.data(), value});
        REQUIRE(status[0]);

        queue.enqueue(
            onHost::FrameSpec{1u, 1u, exec},
            KernelBundle{AtomicTestKernel<alpaka::onAcc::scope::Device, T>{}, status.data(), value});

        REQUIRE(status[0]);
    }
};

TEMPLATE_LIST_TEST_CASE("atomicOperationsWorking", "[atomic]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    // According to the CUDA 12.1 Programming Guide, Section 7.14. Atomic Functions, an atomic function performs a
    // read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory.
    // Some operations require a compute capability of 5.0, 6.0, or higher; on older devices they can be emulated with
    // an atomicCAS loop.

    // According to SYCL 2020 rev. 7, Section 4.15.3. Atomic references, the template parameter T must be one of the
    // following types:
    //   - int, unsigned int,
    //   - long, unsigned long,
    //   - long long, unsigned long long,
    //   - float, or double.
    // In addition, the type T must satisfy one of the following conditions:
    //  - sizeof(T) == 4, or
    //  - sizeof(T) == 8 and the code containing the atomic_ref was submitted to a device that has aspect::atomic64.

    TestAtomicOperations<unsigned int>::testAtomicOperations(device, exec);
    TestAtomicOperations<int>::testAtomicOperations(device, exec);

    TestAtomicOperations<unsigned long>::testAtomicOperations(device, exec);
    TestAtomicOperations<long>::testAtomicOperations(device, exec);

    TestAtomicOperations<unsigned long long>::testAtomicOperations(device, exec);
    TestAtomicOperations<long long>::testAtomicOperations(device, exec);

    TestAtomicOperations<float>::testAtomicOperations(device, exec);
    TestAtomicOperations<double>::testAtomicOperations(device, exec);
}
