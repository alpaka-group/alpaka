/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/engine/philox/PhiloxBaseCommon.hpp"
#include "alpaka/rand/engine/philox/multiplyAndSplit64to32.hpp"

namespace alpaka::rand::engine::internal
{
    /** Philox engine generating a vector of numbers
     *
     * This engine's operator() will return a vector of numbers corresponding to the full size of its counter.
     * This is a convenience vs. memory size tradeoff since the user has to deal with the output array
     * themselves, but the internal state comprises only of a single counter and a key.
     *
     * @tparam T_Params Basic parameters for the Philox algorithm
     */
    template<typename T_Params>
    class PhiloxVector : public PhiloxBaseCommon<T_Params, PhiloxVector<T_Params>>
    {
    public:
        using Base = PhiloxBaseCommon<T_Params, PhiloxVector<T_Params>>;

        /// Counter type
        using Counter = typename Base::Counter;
        /// Key type
        using Key = typename Base::Key;
        using State = PhiloxState<Counter, Key, PhiloxVector<T_Params>>;
        template<typename T_DistributionResultScalar>
        using ResultContainer = typename Base::template ResultContainer<T_DistributionResultScalar>;

    protected:
        /** Get the next array of random numbers and advance internal state
         *
         * @return The next array of random numbers
         */
        constexpr auto nextVector()
        {
            this->advanceCounter(this->state.counter);
            return this->nRounds(this->state.counter, this->state.key);
        }

        /** Skips the next \a offset vectors
         *
         * Unlike its counterpart in \a PhiloxSingle, this function advances the state in multiples of the
         * counter size thus skipping the entire array of numbers.
         */
        constexpr void skip(uint64_t offset)
        {
            this->skip4(offset);
        }

    public:
        /** Construct a new Philox engine with vector output
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset numbers from the start of the subsequence
         */
        // same parameter order as curand_init(seed, subsequence, offset)
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr explicit PhiloxVector(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
            : Base(State{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}})
        {
            this->skipSubsequence(subsequence);
            skip(offset);
            nextVector();
        }

        /** Get the next vector of random numbers
         *
         * @return The next vector of random numbers
         */
        constexpr auto operator()()
        {
            return nextVector();
        }
    };
} // namespace alpaka::rand::engine::internal
