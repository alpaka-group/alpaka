/* Copyright 2022 Luca Ferragina
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/rand/Traits.hpp>

// Backend specific imports.
#    if defined(ALPAKA_ACC_SYCL_ENABLED)
#        include <CL/sycl.hpp>
#        include <oneapi/mkl/rng.hpp>

#        include <oneapi/dpl/random>
#    endif

#    include <type_traits>

namespace alpaka::rand
{
    //! The SYCL rand implementation.
    class RandGenericSycl : public concepts::Implements<ConceptRand, RandGenericSycl>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

    namespace distribution::sycl_rand
    {
        //! The SYCL random number floating point normal distribution.
        template<typename T>
        class NormalReal;

        //! The SYCL random number floating point uniform distribution.
        template<typename T>
        class UniformReal;

        //! The SYCL random number integer uniform distribution.
        template<typename T>
        class UniformUint;
    } // namespace distribution::sycl_rand

    namespace engine::sycl_rand
    {
        //! The SYCL linear congruential random number generator engine.
        class Minstd
        {
        public:
            // After calling this constructor the instance is not valid initialized and
            // need to be overwritten with a valid object
            Minstd() = default;

            Minstd(std::uint32_t const& seed, std::uint32_t const& subsequence = 0, std::uint32_t const& offset = 0)
            {
                oneapi::dpl::minstd_rand engine(seed, offset);
                rng_engine = engine;
            }

        private:
            template<typename T>
            friend class distribution::sycl_rand::NormalReal;
            template<typename T>
            friend class distribution::sycl_rand::UniformReal;
            template<typename T>
            friend class distribution::sycl_rand::UniformUint;

            oneapi::dpl::minstd_rand rng_engine;

        public:
            using result_type = decltype(rng_engine);

            // ALPAKA_FN_HOST_ACC constexpr static result_type min()
            // {
            //     return std::numeric_limits<result_type>::min();
            // }
            // ALPAKA_FN_HOST_ACC constexpr static result_type max()
            // {
            //     return std::numeric_limits<result_type>::max();
            // }
            result_type operator()()
            {
                return rng_engine;
            }
        };
    } // namespace engine::sycl_rand

    namespace distribution::sycl_rand
    {
        //! The SYCL random number float normal distribution.
        template<>
        class NormalReal<float>
        {
        public:
            template<typename TEngine>
            auto operator()(TEngine& engine) -> float
            {
                // Create float uniform_real_distribution distribution
                oneapi::dpl::normal_distribution<float> distr;

                // Generate float random number
                return distr(engine);
            }
        };

        //! The SYCL random number double normal distribution.
        template<>
        class NormalReal<double>
        {
        public:
            template<typename TEngine>
            auto operator()(TEngine& engine) -> double
            {
                // Create float uniform_real_distribution distribution
                oneapi::dpl::normal_distribution<double> distr;

                // Generate float random number
                return distr(engine);
            }
        };

        //! The SYCL random number float uniform distribution.
        template<>
        class UniformReal<float>
        {
        public:
            template<typename TEngine>
            auto operator()(TEngine& engine) -> float
            {
                // Create float uniform_real_distribution distribution
                oneapi::dpl::uniform_real_distribution<float> distr;

                // Generate float random number
                return distr(engine);
                // NOTE: (1.0f - curand_uniform) does not work, because curand_uniform seems to return
                // denormalized floats around 0.f. [0.f, 1.0f)
                // return fUniformRand * static_cast<float>(fUniformRand != 1.0f);
            }
        };

        //! The SYCL random number float uniform distribution.
        template<>
        class UniformReal<double>
        {
        public:
            template<typename TEngine>
            auto operator()(TEngine& engine) -> double
            {
                // Create float uniform_real_distribution distribution
                oneapi::dpl::uniform_real_distribution<double> distr;

                // Generate float random number
                return distr(engine);
                // NOTE: (1.0f - curand_uniform_double) does not work, because curand_uniform_double seems to
                // return denormalized floats around 0.f. [0.f, 1.0f)
                // return fUniformRand * static_cast<double>(fUniformRand != 1.0);
            }
        };

        //! The SYCL random number unsigned integer uniform distribution.
        template<>
        class UniformUint<unsigned int>
        {
        public:
            template<typename TEngine>
            auto operator()(TEngine& engine) -> unsigned int
            {
                // Create float uniform_real_distribution distribution
                oneapi::dpl::uniform_int_distribution<unsigned int> distr;

                // Generate float random number
                return distr(engine);
            }
        };
    } // namespace distribution::sycl_rand

    namespace distribution::trait
    {
        //! The SYCL random number float normal distribution get trait specialization.
        template<typename T>
        struct CreateNormalReal<RandGenericSycl, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static auto createNormalReal(RandGenericSycl const& /*rand*/) -> sycl_rand::NormalReal<T>
            {
                return {};
            }
        };

        //! The SYCL random number float uniform distribution get trait specialization.
        template<typename T>
        struct CreateUniformReal<RandGenericSycl, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static auto createUniformReal(RandGenericSycl const& /*rand*/) -> sycl_rand::UniformReal<T>
            {
                return {};
            }
        };

        //! The SYCL random number integer uniform distribution get trait specialization.
        template<typename T>
        struct CreateUniformUint<RandGenericSycl, T, std::enable_if_t<std::is_integral_v<T>>>
        {
            static auto createUniformUint(RandGenericSycl const& /*rand*/) -> sycl_rand::UniformUint<T>
            {
                return {};
            }
        };
    } // namespace distribution::trait

    namespace engine::trait
    {
        //! The SYCL random number default generator get trait specialization.
        template<>
        struct CreateDefault<RandGenericSycl>
        {
            static auto createDefault(
                RandGenericSycl const& /*rand*/,
                std::uint32_t const& seed = 0,
                std::uint32_t const& subsequence = 0,
                std::uint32_t const& offset = 0) -> sycl_rand::Minstd
            {
                return {seed, subsequence, offset};
            }
        };
    } // namespace engine::trait
#    endif
} // namespace alpaka::rand

#endif
