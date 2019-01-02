/**
 * \file
 * Copyright 2017 Rene Widera, Benjamin Worpitz
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

// NVCC needs --expt-relaxed-constexpr
#if !defined(__NVCC__) || \
    ( defined(__NVCC__) && defined(__CUDACC_RELAXED_CONSTEXPR__) )

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

#include <limits>

//#############################################################################
//!
//#############################################################################
class KernelWithHostConstexpr
{
public:
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool* success) const
    -> void
    {
        alpaka::ignore_unused(acc);

#if BOOST_COMP_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4127)  // warning C4127: conditional expression is constant
#endif
        // FIXME: workaround for HIP(HCC) where numeric_limits::* do not provide
        // matching host-device restriction requirements
#if defined(BOOST_COMP_HCC) && BOOST_COMP_HCC
        constexpr auto max = static_cast<std::uint32_t>(-1);
#else
        constexpr auto max = std::numeric_limits< std::uint32_t >::max();
#endif
        ALPAKA_CHECK(*success, 0 != max);
#if BOOST_COMP_MSVC
    #pragma warning(pop)
#endif
    }
};

//-----------------------------------------------------------------------------
//
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

    KernelWithHostConstexpr kernel;

    REQUIRE(fixture(kernel));
}
};

TEST_CASE( "kernelWithHostConstexpr", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}

#endif
