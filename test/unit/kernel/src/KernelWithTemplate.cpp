/**
 * \file
 * Copyright 2017-2019 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

#include <type_traits>


//#############################################################################
template<
    typename T>
class KernelFuntionObjectTemplate
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<std::int32_t, T>::value,
            "Incorrect additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelFuntionObjectTemplate<std::int32_t> kernel;

    REQUIRE(fixture(kernel));
}
};

//#############################################################################
class KernelInvocationWithAdditionalTemplate
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
        T const &) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<std::int32_t, T>::value,
            "Incorrect additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
struct TestTemplateExtra
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelInvocationWithAdditionalTemplate kernel;

    REQUIRE(fixture(kernel, std::int32_t()));
}
};

TEST_CASE( "kernelFuntionObjectTemplate", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}

TEST_CASE( "kernelFuntionObjectExtraTemplate", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateExtra() );
}
