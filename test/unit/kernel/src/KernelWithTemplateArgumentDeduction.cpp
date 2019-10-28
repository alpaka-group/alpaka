/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera, Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/kernel/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

#include <type_traits>

//#############################################################################
template< typename TExpected >
class KernelInvocationTemplateDeductionValueSemantics
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAcc,
        typename TByValue,
        typename TByConstValue,
        typename TByConstReference>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            bool * success,
            TByValue,
            TByConstValue const,
            TByConstReference const &) const
        -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<TByValue, TExpected>::value,
            "Incorrect first additional kernel template parameter type!");
        static_assert(
            std::is_same<TByConstValue, TExpected>::value,
            "Incorrect second additional kernel template parameter type!");
        static_assert(
            std::is_same<TByConstReference, TExpected>::value,
            "Incorrect third additional kernel template parameter type!");

    }
};

//-----------------------------------------------------------------------------
struct TestTemplateDeductionFromValue
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        using Value = std::int32_t;
        KernelInvocationTemplateDeductionValueSemantics< Value > kernel;

        Value value{ };
        REQUIRE(fixture(kernel, value, value, value));
    }
};

struct TestTemplateDeductionFromConstValue
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        using Value = std::int32_t;
        KernelInvocationTemplateDeductionValueSemantics< Value > kernel;

        Value const constValue{ };
        REQUIRE(fixture(kernel, constValue, constValue, constValue));
    }
};

struct TestTemplateDeductionFromConstReference
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        using Value = std::int32_t;
        KernelInvocationTemplateDeductionValueSemantics< Value > kernel;

        Value value{ };
        Value const & constReference = value;
        REQUIRE(fixture(kernel, constReference, constReference, constReference));
    }
};

//#############################################################################
template<
    typename TExpectedFirst,
    typename TExpectedSecond = TExpectedFirst
>
class KernelInvocationTemplateDeductionPointerSemantics
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAcc,
        typename TByPointer,
        typename TByPointerToConst>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            bool * success,
            TByPointer *,
            TByPointerToConst const *) const
        -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<TByPointer, TExpectedFirst>::value,
            "Incorrect first additional kernel template parameter type!");
        static_assert(
            std::is_same<TByPointerToConst, TExpectedSecond>::value,
            "Incorrect second additional kernel template parameter type!");

    }
};

//-----------------------------------------------------------------------------
struct TestTemplateDeductionFromPointer
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        using Value = std::int32_t;
        KernelInvocationTemplateDeductionPointerSemantics< Value > kernel;

        Value value{ };
        Value * pointer = &value;
        REQUIRE(fixture(kernel, pointer, pointer));
    }
};

struct TestTemplateDeductionFromPointerToConst
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        using Value = std::int32_t;
        KernelInvocationTemplateDeductionPointerSemantics< Value const, Value > kernel;

        Value const constValue{ };
        Value const * pointerToConst = &constValue;
        REQUIRE(fixture(kernel, pointerToConst, pointerToConst));
    }
};

struct TestTemplateDeductionFromStaticArray
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        using Value = std::int32_t;
        KernelInvocationTemplateDeductionPointerSemantics< Value > kernel;

        Value staticArray[4] = { };
        REQUIRE(fixture(kernel, staticArray, staticArray));
    }
};

struct TestTemplateDeductionFromConstStaticArray
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        using Value = std::int32_t;
        KernelInvocationTemplateDeductionPointerSemantics< Value const, Value > kernel;

        Value const constStaticArray[4] = { };
        REQUIRE(fixture(kernel, constStaticArray, constStaticArray));
    }
};

TEST_CASE( "kernelFuntionObjectTemplateDeductionFromValue", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDeductionFromValue() );
}

TEST_CASE( "kernelFuntionObjectTemplateDeductionFromConstValue", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDeductionFromConstValue() );
}

TEST_CASE( "kernelFuntionObjectTemplateDeductionFromConstReference", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDeductionFromConstReference() );
}

TEST_CASE( "kernelFuntionObjectTemplateDeductionFromPointer", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDeductionFromPointer() );
}

TEST_CASE( "kernelFuntionObjectTemplateDeductionFromPointerToConst", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDeductionFromPointerToConst() );
}

TEST_CASE( "kernelFuntionObjectTemplateDeductionFromStaticArray", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDeductionFromStaticArray() );
}

TEST_CASE( "kernelFuntionObjectTemplateDeductionFromConstStaticArray", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDeductionFromConstStaticArray() );
}
