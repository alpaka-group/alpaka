/**
 * \file
 * Copyright 2017-2018 Benjamin Worpitz, Axel Huebl
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

#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

//#############################################################################
class RandTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T_Generator
    >
    ALPAKA_FN_ACC void
    genNumbers(
        TAcc const & acc,
        bool * success,
        T_Generator & gen
    ) const
    {
        {
            auto dist(alpaka::rand::distribution::createNormalReal<float>(acc));
            auto const r = dist(gen);
#if !BOOST_ARCH_PTX
            ALPAKA_CHECK(*success, std::isfinite(r));
#else
            alpaka::ignore_unused(r);
#endif
        }

        {
            auto dist(alpaka::rand::distribution::createNormalReal<double>(acc));
            auto const r = dist(gen);
#if !BOOST_ARCH_PTX
            ALPAKA_CHECK(*success, std::isfinite(r));
#else
            alpaka::ignore_unused(r);
#endif
        }
        {
            auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));
            auto const r = dist(gen);
            ALPAKA_CHECK(*success, 0.0f <= r);
            ALPAKA_CHECK(*success, 1.0f > r);
        }

        {
            auto dist(alpaka::rand::distribution::createUniformReal<double>(acc));
            auto const r = dist(gen);
            ALPAKA_CHECK(*success, 0.0 <= r);
            ALPAKA_CHECK(*success, 1.0 > r);
        }

        {
            auto dist(alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc));
            auto const r = dist(gen);
            alpaka::ignore_unused(r);
        }
    }

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
        // default generator for accelerator
        auto genDefault = alpaka::rand::generator::createDefault(
            acc,
            12345u,
            6789u
        );
        genNumbers( acc, success, genDefault );

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && \
  !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // std::random_device
        auto genRandomDevice = alpaka::rand::generator::createDefault(
            alpaka::rand::RandomDevice{},
            12345u,
            6789u
        );
        genNumbers( acc, success, genRandomDevice );

        // MersenneTwister
        auto genMersenneTwister = alpaka::rand::generator::createDefault(
            alpaka::rand::MersenneTwister{},
            12345u,
            6789u
        );
        genNumbers( acc, success, genMersenneTwister );

        // TinyMersenneTwister
        auto genTinyMersenneTwister = alpaka::rand::generator::createDefault(
            alpaka::rand::TinyMersenneTwister{},
            12345u,
            6789u
        );
        genNumbers( acc, success, genTinyMersenneTwister );
#endif
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

    RandTestKernel kernel;

    REQUIRE(
        fixture(
            kernel));
}
};

TEST_CASE( "defaultRandomGeneratorIsWorking", "[rand]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}
