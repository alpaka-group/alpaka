/* Copyright 2025 Mehmet Yusufoglu, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/math.hpp"
#include "alpaka/rand/concepts.hpp"
#include "alpaka/rand/distribution/UniformReal.hpp"

#include <cmath>

namespace alpaka::rand::distribution
{
    namespace internal
    {
        /**
         * @brief Scalar Box-Muller transform for floating-point types.
         *
         * Generates independent standard normal samples from a uniform
         * scalar engine. The implementation caches the second sample so
         * that a pair of uniforms yields two normal values.
         *
         * Ref.: G. E. P. Box and M. E. Muller, "A Note on the Generation of Random Normal Deviates,"
         */
        template<std::floating_point T_Fp>
        struct BoxMuller
        {
            using ResultType = T_Fp;

            template<concepts::UniformStdEngine T_Rng>
            constexpr T_Fp operator()(T_Rng& rng)
            {
                // Re-use cached second sample
                if(m_hasSecondRngNumber)
                {
                    m_hasSecondRngNumber = false;
                    return m_secondRngNumber;
                }
                // Generate two uniform floats in (0,1)
                constexpr auto uniformDist = UniformReal{T_Fp{0}, T_Fp{1}, interval::oo};

                T_Fp u1 = uniformDist(rng);
                T_Fp u2 = uniformDist(rng);

                // Box-Muller transform
                T_Fp r = math::sqrt(T_Fp{-2} * math::log(u1));
                T_Fp theta = T_Fp{2} * static_cast<T_Fp>(M_PI) * u2;
                T_Fp z0 = r * math::cos(theta);
                T_Fp z1 = r * math::sin(theta);

                m_secondRngNumber = z1;
                m_hasSecondRngNumber = true;
                return z0;
            }

        private:
            T_Fp m_secondRngNumber{};
            bool m_hasSecondRngNumber = false;
        };
    } // namespace internal

    /**
     * @brief Used to sample floating-point values from a normal(-/gaussian) distribution.
     *
     * Usage is analogous to std::normal_distribution<T_Result> @see
     * https://en.cppreference.com/w/cpp/numeric/random/normal_distribution.html
     * -> Generates N(mean, stddev).
     * Is using the Box-Muller method of generating normally distributed random numbers from a uniform distribution.
     *
     * @note: currently supports 32 and 64 bit floating point types.
     */
    template<std::floating_point T_Result>
    struct NormalReal
    {
        using ResultType = T_Result;

        /**
         * @brief Constructs normal(-/gaussian) distribution with given parameters.
         *
         * @param mean   Mean of the target normal distribution.
         * @param stdDev Standard deviation of the target normal distribution.
         *
         * The default is N(0,1). The distribution is sampled using the Box-Muller method.
         * This implementation keeps an internal state, therefore each
         * thread/worker must use its own instance to avoid data races
         * when the same object is accessed concurrently by multiple
         * workers.
         *
         * Usage is otherwise analogous to std::normal_distribution<T_Result> @see
         * https://en.cppreference.com/w/cpp/numeric/random/normal_distribution.html
         */
        constexpr explicit NormalReal(T_Result mean = T_Result{0}, T_Result stdDev = T_Result{1})
            : m_mean(mean)
            , m_stdDev(stdDev)
        {
        }

        /** Selects a value from a normal (-/gaussian) distribution for the configured (mean,stdDev) settings.
         *
         * @param engine a random engine conforming to the `UniformRandomEngine` concept
         * (currently accepts stdlib uniform engines and alpaka engines included in the alpaka::rand::engine namespace)
         * @return a floating-point value sampled from the configured distribution.
         */
        template<concepts::UniformStdEngine T_Engine>
        constexpr ResultType operator()(T_Engine& engine)
        {
            return m_impl(engine) * m_stdDev + m_mean;
        }

    private:
        // current implementation
        using Impl = internal::BoxMuller<T_Result>;
        // box muller has a state and must therefore be an accessible field.
        Impl m_impl;
        ResultType const m_mean;
        ResultType const m_stdDev;
    };

} // namespace alpaka::rand::distribution
