/* Copyright 2023 Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/rand/Traits.hpp>

// Backend specific imports.
#    include <CL/sycl.hpp>
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wcast-align"
#        pragma clang diagnostic ignored "-Wcast-qual"
#        pragma clang diagnostic ignored "-Wextra-semi"
#        pragma clang diagnostic ignored "-Wfloat-equal"
#        pragma clang diagnostic ignored "-Wold-style-cast"
#        pragma clang diagnostic ignored "-Wreserved-identifier"
#        pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#        pragma clang diagnostic ignored "-Wsign-compare"
#        pragma clang diagnostic ignored "-Wundef"
#    endif
#    include <oneapi/mkl/rng.hpp>

#    include <oneapi/dpl/random>
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif

#    include <type_traits>

namespace alpaka::rand
{
    //! The SYCL rand implementation.
    template<typename TDim>
    class RandGenericSycl : public concepts::Implements<ConceptRand, RandGenericSycl<TDim>>
    {
    public:
        RandGenericSycl(sycl::nd_item<TDim::value> my_item) : m_item{my_item}
        {
        }

        sycl::nd_item<TDim::value> m_item;
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
        template<typename TDim>
        class Minstd
        {
        public:
            // After calling this constructor the instance is not valid initialized and
            // need to be overwritten with a valid object
            Minstd() = default;

            Minstd(RandGenericSycl<TDim> rand, std::uint32_t const& seed)
            {
                oneapi::dpl::minstd_rand engine(seed, rand.m_item.get_global_linear_id());
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
            using result_type = float;

            ALPAKA_FN_HOST_ACC static result_type min()
            {
                return std::numeric_limits<result_type>::min();
            }
            ALPAKA_FN_HOST_ACC static result_type max()
            {
                return std::numeric_limits<result_type>::max();
            }
            result_type operator()()
            {
                oneapi::dpl::uniform_real_distribution<float> distr;
                return distr(rng_engine);
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
                return distr(engine.rng_engine);
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
                return distr(engine.rng_engine);
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
                return distr(engine.rng_engine);
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
                return distr(engine.rng_engine);
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
                return distr(engine.rng_engine);
            }
        };
    } // namespace distribution::sycl_rand

    namespace distribution::trait
    {
        //! The SYCL random number float normal distribution get trait specialization.
        template<typename TDim, typename T>
        struct CreateNormalReal<RandGenericSycl<TDim>, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static auto createNormalReal(RandGenericSycl<TDim> const& /*rand*/) -> sycl_rand::NormalReal<T>
            {
                return {};
            }
        };

        //! The SYCL random number float uniform distribution get trait specialization.
        template<typename TDim, typename T>
        struct CreateUniformReal<RandGenericSycl<TDim>, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static auto createUniformReal(RandGenericSycl<TDim> const& /*rand*/) -> sycl_rand::UniformReal<T>
            {
                return {};
            }
        };

        //! The SYCL random number integer uniform distribution get trait specialization.
        template<typename TDim, typename T>
        struct CreateUniformUint<RandGenericSycl<TDim>, T, std::enable_if_t<std::is_integral_v<T>>>
        {
            static auto createUniformUint(RandGenericSycl<TDim> const& /*rand*/) -> sycl_rand::UniformUint<T>
            {
                return {};
            }
        };
    } // namespace distribution::trait

    namespace engine::trait
    {
        //! The SYCL random number default generator get trait specialization.
        template<typename TDim>
        struct CreateDefault<RandGenericSycl<TDim>>
        {
            static auto createDefault(
                RandGenericSycl<TDim> const& rand,
                std::uint32_t const& seed = 0,
                std::uint32_t const& /* subsequence */ = 0,
                std::uint32_t const& /* offset */ = 0) -> sycl_rand::Minstd<TDim>
            {
                return {rand, seed};
            }
        };
    } // namespace engine::trait
#    endif
} // namespace alpaka::rand

#endif
