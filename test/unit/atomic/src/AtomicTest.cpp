/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/math/FloatEqualExact.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <climits>
#include <type_traits>

template<typename TT1, typename TT2>
ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC bool equals(TT1 a, TT2 b)
{
    return a == b;
}

ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC bool equals(float a, float b)
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC bool equals(double a, double b)
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_add(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    TT const reference = static_cast<TT>(operand_orig + value);
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicAdd(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_sub(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    TT const reference = static_cast<TT>(operand_orig - value);
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicSub>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicSub(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_min(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    TT const reference = (operand_orig < value) ? operand_orig : value;
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicMin>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicMin(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_max(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    TT const reference = (operand_orig > value) ? operand_orig : value;
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicMax>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicMax(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_exch(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    TT const reference = value;
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicExch>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicExch(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_inc(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    // \TODO: Check reset to 0 at 'value'.
    TT const value = static_cast<TT>(42);
    TT const reference = static_cast<TT>(operand_orig + 1);
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicInc>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicInc(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_dec(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    // \TODO: Check reset to 'value' at 0.
    TT const value = static_cast<TT>(42);
    TT const reference = static_cast<TT>(operand_orig - 1);
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicDec>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicDec(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_and(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    TT const reference = operand_orig & value;
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicAnd>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicAnd(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_or(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    TT const reference = operand_orig | value;
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicOr>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOr(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_xor(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(operand_orig + static_cast<TT>(4));
    TT const reference = operand_orig ^ value;
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicOp<alpaka::AtomicXor>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operand_orig;
        TT const ret = alpaka::atomicXor(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operand_orig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TT>
ALPAKA_FN_ACC auto test_atomic_cas(TAcc const& acc, bool* success, TT operand_orig) -> void
{
    TT const value = static_cast<TT>(4);
    auto& operand = alpaka::declareSharedVar<TT, __COUNTER__>(acc);

    // with match
    {
        TT const compare = operand_orig;
        TT const reference = value;
        {
            operand = operand_orig;
            TT const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operand_orig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operand_orig;
            TT const ret = alpaka::atomicCas(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operand_orig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
    }

    // without match
    {
        TT const compare = static_cast<TT>(operand_orig + static_cast<TT>(1));
        TT const reference = operand_orig;
        {
            operand = operand_orig;
            TT const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operand_orig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operand_orig;
            TT const ret = alpaka::atomicCas(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operand_orig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
    }
}

template<typename TAcc, typename TT, typename TSfinae = void>
class AtomicTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, TT operand_orig) const -> void
    {
        test_atomic_add(acc, success, operand_orig);
        test_atomic_sub(acc, success, operand_orig);

        test_atomic_min(acc, success, operand_orig);
        test_atomic_max(acc, success, operand_orig);

        test_atomic_exch(acc, success, operand_orig);

        test_atomic_inc(acc, success, operand_orig);
        test_atomic_dec(acc, success, operand_orig);

        test_atomic_and(acc, success, operand_orig);
        test_atomic_or(acc, success, operand_orig);
        test_atomic_xor(acc, success, operand_orig);

        test_atomic_cas(acc, success, operand_orig);
    }
};

template<typename TAcc, typename TT>
class AtomicTestKernel<TAcc, TT, std::enable_if_t<std::is_floating_point<TT>::value>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, TT operand_orig) const -> void
    {
        test_atomic_add(acc, success, operand_orig);
        test_atomic_sub(acc, success, operand_orig);

        test_atomic_min(acc, success, operand_orig);
        test_atomic_max(acc, success, operand_orig);

        test_atomic_exch(acc, success, operand_orig);

        // These are not supported on float/double types
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);
        // testAtomicCas(acc, success, operandOrig);
    }
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
// Skip all atomic tests for the unified CUDA/HIP backend.
// CUDA and HIP atomics will be tested separate.
template<typename T, typename TDim, typename TIdx>
class AtomicTestKernel<
    alpaka::AccGpuUniformCudaHipRt<TDim, TIdx>,
    T,
    std::enable_if_t<!std::is_floating_point<T>::value>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuUniformCudaHipRt<TDim, TIdx> const& acc, bool* success, T operandOrig)
        const -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(success);
        alpaka::ignore_unused(operandOrig);
    }
};

// We need this partial specialization because of partial ordering of the
// template specializations
template<typename T, typename TDim, typename TIdx>
class AtomicTestKernel<
    alpaka::AccGpuUniformCudaHipRt<TDim, TIdx>,
    T,
    std::enable_if_t<std::is_floating_point<T>::value>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuUniformCudaHipRt<TDim, TIdx> const& acc, bool* success, T operandOrig)
        const -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(success);
        alpaka::ignore_unused(operandOrig);
    }
};
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, int operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

// NOTE: unsigned int is the only type supported by all atomic CUDA operations.
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, unsigned int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, unsigned int operandOrig)
        const -> void
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

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, unsigned long int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuCudaRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicSub(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#        endif
#    endif

        testAtomicExch(acc, success, operandOrig);

#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#        endif
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, unsigned long long int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuCudaRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#    endif

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, float>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, float operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, double>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, double operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        // Not supported
        // testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    alpaka::AccGpuCudaRt<TDim, TIdx>,
    T,
    std::enable_if_t<
        !std::is_same<int, T>::value && !std::is_same<unsigned int, T>::value
        && !std::is_same<unsigned long int, T>::value && !std::is_same<unsigned long long int, T>::value
        && !std::is_same<float, T>::value && !std::is_same<double, T>::value>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, T operandOrig) const
        -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(operandOrig);

        // All other types are not supported by CUDA atomic operations.
        ALPAKA_CHECK(*success, true);
    }
};
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, int operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

// NOTE: unsigned int is the only type supported by all atomic HIP operations.
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, unsigned int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, unsigned int operandOrig)
        const -> void
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

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, unsigned long int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuHipRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicSub(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#        endif
#    endif

        testAtomicExch(acc, success, operandOrig);

#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#        endif
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, unsigned long long int>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuHipRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#    endif

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, float>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, float operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, double>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, double operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        // Not supported
        // testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    alpaka::AccGpuHipRt<TDim, TIdx>,
    T,
    std::enable_if_t<
        !std::is_same<int, T>::value && !std::is_same<unsigned int, T>::value
        && !std::is_same<unsigned long int, T>::value && !std::is_same<unsigned long long int, T>::value
        && !std::is_same<float, T>::value && !std::is_same<double, T>::value>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, T operandOrig) const
        -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(operandOrig);

        // All other types are not supported by HIP atomic operations.
        ALPAKA_CHECK(*success, true);
    }
};
#endif

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED
template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    alpaka::AccOacc<TDim, TIdx>,
    T,
    std::enable_if_t<sizeof(T) <= 2>> // disable 8-bit and 16-bit tests
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccOacc<TDim, TIdx> const& acc, bool* success, T operandOrig) const -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(operandOrig);

        // All other types are not supported by Oacc atomic operations.
        ALPAKA_CHECK(*success, true);
    }
};
#endif

template<typename TAcc, typename TT>
struct TestAtomicOperations
{
    static auto test_atomic_operations() -> void
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

        AtomicTestKernel<TAcc, TT> kernel;

        TT value = static_cast<TT>(32);
        REQUIRE(fixture(kernel, value));
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("atomicOperationsWorking", "[atomic]", TestAccs)
{
    using Acc = TestType;

    TestAtomicOperations<Acc, unsigned char>::test_atomic_operations();
    TestAtomicOperations<Acc, char>::test_atomic_operations();
    TestAtomicOperations<Acc, unsigned short>::test_atomic_operations();
    TestAtomicOperations<Acc, short>::test_atomic_operations();

    TestAtomicOperations<Acc, unsigned int>::test_atomic_operations();
    TestAtomicOperations<Acc, int>::test_atomic_operations();

    TestAtomicOperations<Acc, unsigned long>::test_atomic_operations();
    TestAtomicOperations<Acc, long>::test_atomic_operations();
    TestAtomicOperations<Acc, unsigned long long>::test_atomic_operations();
    TestAtomicOperations<Acc, long long>::test_atomic_operations();

    TestAtomicOperations<Acc, float>::test_atomic_operations();
    TestAtomicOperations<Acc, double>::test_atomic_operations();
}
