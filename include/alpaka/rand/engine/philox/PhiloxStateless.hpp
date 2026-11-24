/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/rand/engine/philox/PhiloxConstants.hpp"
#include "alpaka/rand/engine/philox/multiplyAndSplit64to32.hpp"

#include <utility>

namespace alpaka::rand::engine::internal
{
    /** Philox algorithm parameters
     *
     * @tparam T_counterSize number of elements in the counter
     * @tparam T_width width of one counter element (in bits)
     * @tparam T_rounds number of S-box rounds
     */
    template<unsigned T_counterSize, unsigned T_width, unsigned T_rounds>
    struct PhiloxParams
    {
        static constexpr unsigned counterSize = T_counterSize;
        static constexpr unsigned width = T_width;
        static constexpr unsigned rounds = T_rounds;
    };

    /** Class basic Philox family counter-based PRNG
     *
     * Checks the validity of passed-in parameters and calls the backend methods to perform N rounds of the
     * Philox shuffle.
     *
     * @tparam T_Params Philox algorithm parameters \sa PhiloxParams
     */
    template<typename T_Params>
    class PhiloxStateless
    {
        static constexpr unsigned numRounds()
        {
            return T_Params::rounds;
        }

        static constexpr unsigned vectorSize()
        {
            return T_Params::counterSize;
        }

        static constexpr unsigned numberWidth()
        {
            return T_Params::width;
        }

        static_assert(numRounds() > 0, "Number of Philox rounds must be > 0.");
        static_assert(vectorSize() % 2 == 0, "Philox counter size must be an even number.");
        static_assert(vectorSize() <= 16, "Philox SP network is not specified for sizes > 16.");
        static_assert(numberWidth() % 8 == 0, "Philox number width in bits must be a multiple of 8.");

        static_assert(numberWidth() == 32, "Philox implemented only for 32 bit numbers.");

    public:
        using Counter = alpaka::Vec<std::uint32_t, T_Params::counterSize>;
        using Key = alpaka::Vec<std::uint32_t, T_Params::counterSize / 2>;

    protected:
        /** Single round of the Philox shuffle
         *
         * @param counter state of the counter
         * @param key value of the key
         * @return shuffled counter
         */
        static constexpr auto singleRound(Counter const& counter, Key const& key)
        {
            std::uint32_t h0, l0, h1, l1;
            multiplyAndSplit64to32(counter[0], PhiloxConstants::multipliter4x32p0(), h0, l0);
            multiplyAndSplit64to32(counter[2], PhiloxConstants::multipliter4x32p1(), h1, l1);
            return Counter{h1 ^ counter[1] ^ key[0], l1, h0 ^ counter[3] ^ key[1], l0};
        }

        /** Bump the \a key by the Weyl sequence step parameter
         *
         * @param key the key to be bumped
         * @return the bumped key
         */
        static constexpr auto bumpKey(Key const& key)
        {
            return Key{key[0] + PhiloxConstants::weyl32p0(), key[1] + PhiloxConstants::weyl32p1()};
        }

        /** Performs N rounds of the Philox shuffle
         *
         * @param counterIn initial state of the counter
         * @param keyIn initial state of the key
         * @return result of the PRNG shuffle; has the same size as the counter
         */
        static constexpr auto nRounds(Counter const& counterIn, Key const& keyIn) -> Counter
        {
            Key key{keyIn};
            Counter counter = singleRound(counterIn, key);

            // Use a constexpr variable to ensure the unroll factor is a compile-time constant
            constexpr unsigned rounds = numRounds();

            // The call above is round one, so the loop adds rounds - 1 further rounds.
            for(unsigned int n = 1; n < rounds; ++n)
            {
                key = bumpKey(key);
                counter = singleRound(counter, key);
            }

            return counter;
        }

    public:
        /** Generates a random number (\p TCounterSize x32-bit)
         *
         * @param counter initial state of the counter
         * @param key initial state of the key
         * @return result of the PRNG shuffle; has the same size as the counter
         */
        static constexpr auto generate(Counter const& counter, Key const& key) -> Counter
        {
            return nRounds(counter, key);
        }
    };
} // namespace alpaka::rand::engine::internal
