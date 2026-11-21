/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/MultiplyAndSplit64to32.hpp"
#include "alpaka/rand/Philox/PhiloxBaseCommon.hpp"

#include <utility>

namespace alpaka::rand::engine
{
    /** Philox state for vector generator
     *
     * @tparam TCounter Type of the Counter array
     * @tparam TKey Type of the Key array
     */
    template<typename TCounter, typename TKey>
    struct PhiloxStateVector
    {
        using Counter = TCounter;
        using Key = TKey;

        /// Counter array
        Counter counter;
        /// Key array
        Key key;
    };

    /** Philox engine generating a vector of numbers
     *
     * This engine's operator() will return a vector of numbers corresponding to the full size of its counter.
     * This is a convenience vs. memory size tradeoff since the user has to deal with the output array
     * themselves, but the internal state comprises only of a single counter and a key.
     *
     * @tparam TParams Basic parameters for the Philox algorithm
     */
    template<typename TParams>
    class PhiloxVector : public PhiloxBaseCommon<TParams, PhiloxVector<TParams>>
    {
    public:
        using Base = PhiloxBaseCommon<TParams, PhiloxVector<TParams>>;

        /// Counter type
        using Counter = typename Base::Counter;
        /// Key type
        using Key = typename Base::Key;
        /// State type
        using State = PhiloxStateVector<Counter, Key>;

        template<typename TDistributionResultScalar>
        using ResultContainer = typename Base::template ResultContainer<TDistributionResultScalar>;

        State state;

    protected:
        /** Get the next array of random numbers and advance internal state
         *
         * @return The next array of random numbers
         */
        ALPAKA_FN_HOST_ACC auto nextVector()
        {
            this->advanceCounter(state.counter);
            return this->nRounds(state.counter, state.key);
        }

        /** Skips the next \a offset vectors
         *
         * Unlike its counterpart in \a PhiloxSingle, this function advances the state in multiples of the
         * counter size thus skipping the entire array of numbers.
         */
        ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
        {
            this->skip4(offset);
        }

    public:
        /** Construct a new Philox engine with vector output
         *
         * Vector i is the same as numbers 4 (offset + i) to 4 (offset + i) + 3 of the single-number engine with
         * the same seed and subsequence.
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset vectors from the start of the subsequence
         */
        ALPAKA_FN_HOST_ACC PhiloxVector(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
            : state{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}}
        {
            this->skipSubsequence(subsequence);
            skip(offset);
        }

        /** Get the next vector of random numbers
         *
         * @return The next vector of random numbers
         */
        ALPAKA_FN_HOST_ACC auto operator()()
        {
            return nextVector();
        }
    };
} // namespace alpaka::rand::engine
