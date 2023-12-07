/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jiří Vyskočil <jiri@vyskocil.com>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/TinyMT/tinymt32.h"

#include <cstdint>

namespace alpaka::rand::engine::cpu
{
    //! Implementation of std::UniformRandomBitGenerator for TinyMT32
    struct TinyMTengine
    {
        using result_type = std::uint32_t;

        static constexpr auto default_seed() -> result_type
        {
            return 42u;
        }

        void seed(result_type value = default_seed())
        {
            // parameters from TinyMT/jump/sample.c
            prng.mat1 = 0x8f70'11ee;
            prng.mat2 = 0xfc78'ff1f;
            prng.tmat = 0x3793'fdff;

            tinymt32_init(&prng, value);
        }

        TinyMTengine(std::uint32_t const& seedValue)
        {
            seed(seedValue);
        }

        TinyMTengine()
        {
            seed(default_seed());
        }

        auto operator()() -> result_type
        {
            return tinymt32_generate_uint32(&prng);
        }

        static constexpr auto min() -> result_type
        {
            return 0u;
        }

        static constexpr auto max() -> result_type
        {
            return UINT32_MAX;
        }

        void discard(unsigned long long) // z
        {
            // not implemented
            // tinymt32_jump( &prng, z, z );
        }

        tinymt32_t prng;
    };
} // namespace alpaka::rand::engine::cpu
