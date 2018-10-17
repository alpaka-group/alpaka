/**
 * \file
 * Copyright 2016-2018 Benjamin Worpitz
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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicAdd(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Add>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig + value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicSub(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Sub>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig - value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicMin(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Min>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = (operandOrig < value) ? operandOrig : value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicMax(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Max>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = (operandOrig > value) ? operandOrig : value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicExch(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Exch>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicInc(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    // \TODO: Check reset to 0 at 'value'.
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(42);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Inc>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig + 1;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicDec(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    // \TODO: Check reset to 'value' at 0.
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(42);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Dec>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig - 1;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicAnd(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::And>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig & value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicOr(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Or>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig | value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicXor(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = operandOrig + static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Xor>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig ^ value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicCas(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);

    //-----------------------------------------------------------------------------
    // with match
    {
        operand = operandOrig;
        T const compare = operandOrig;
        T const value = static_cast<T>(4);
        T const ret =
            alpaka::atomic::atomicOp<
                alpaka::atomic::op::Cas>(
                    acc,
                    &operand,
                    compare,
                    value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        T const reference = value;
        ALPAKA_CHECK(*success, operand == reference);
    }

    //-----------------------------------------------------------------------------
    // without match
    {
        operand = operandOrig;
        T const compare = operandOrig + static_cast<T>(1);
        T const value = static_cast<T>(4);
        T const ret =
            alpaka::atomic::atomicOp<
                alpaka::atomic::op::Cas>(
                    acc,
                    &operand,
                    compare,
                    value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        T const reference = operandOrig;
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//#############################################################################
template<
    typename TAcc,
    typename T,
    typename Sfinae = void>
class AtomicTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        T operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
//#############################################################################
// NOTE: std::uint32_t is the only type supported by all atomic CUDA operations.
template<
    typename TDim,
    typename TIdx,
    typename T>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    T,
    typename std::enable_if<std::is_same<std::uint32_t, T>::value>::type>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        T operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx,
    typename T>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    T,
    typename std::enable_if<!std::is_same<std::uint32_t, T>::value>::type>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        T operandOrig) const
    -> void
    {
        // All other types are not supported by all CUDA atomic operations.
        ALPAKA_CHECK(*success, true);
    }
};
#endif

BOOST_AUTO_TEST_SUITE(atomic)

//#############################################################################
template<
    typename TAcc,
    typename T>
struct TestAtomicOperations
{
    //-----------------------------------------------------------------------------
    static auto testAtomicOperations()
    -> void
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        AtomicTestKernel<TAcc, T> kernel;

        T value = static_cast<T>(32);
        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel,
                value));
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    atomicOperationsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    // This test exceeds the maximum compilation time.
#if !defined(ALPAKA_CI)
    TestAtomicOperations<TAcc, std::int8_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::uint8_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::int16_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::uint16_t>::testAtomicOperations();
#endif
    TestAtomicOperations<TAcc, std::int32_t>::testAtomicOperations();
    // The std::uint32_t test is the only one that is compiled for CUDA
    // because this is the only type which is supported by all atomic operations.
    TestAtomicOperations<TAcc, std::uint32_t>::testAtomicOperations();
#if !defined(ALPAKA_CI)
    TestAtomicOperations<TAcc, std::int64_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::uint64_t>::testAtomicOperations();
#endif
    // Not all atomic operations are possible with floating point values.
    //TestAtomicOperations<TAcc, float>::testAtomicOperations();
    //TestAtomicOperations<TAcc, double>::testAtomicOperations();
}

BOOST_AUTO_TEST_SUITE_END()
