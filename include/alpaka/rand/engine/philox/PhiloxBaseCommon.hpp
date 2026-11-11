/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/engine/philox/PhiloxState.hpp"
#include "alpaka/rand/engine/philox/PhiloxStateless.hpp"

namespace alpaka::rand::engine::internal
{
    /** Common class for Philox family engines
     *
     * Relies on `PhiloxStateless` to provide the PRNG and adds state to handling the counting.
     *
     * @tparam T_Params Philox algorithm parameters \sa PhiloxParams
     * @tparam T_Impl engine type implementation (CRTP)
     *
     * static const data members are transformed into functions, because GCC
     * assumes types with static data members to be not mappable and makes not
     * exception for constexpr ones. This is a valid interpretation of the
     * OpenMP <= 4.5 standard. In OpenMP >= 5.0 types with any kind of static
     * data member are mappable.
     */
    template<typename T_Params, typename T_Impl>
    class PhiloxBaseCommon : public PhiloxStateless<T_Params>
    {
    public:
        using Counter = typename PhiloxStateless<T_Params>::Counter;
        using Key = typename PhiloxStateless<T_Params>::Key;
        /// State type
        using State = PhiloxState<Counter, Key, T_Impl>;

        /// Internal engine state
        State state;
        /// Distribution container type
        template<typename T_DistributionResultScalar>
        using ResultContainer = Vec<T_DistributionResultScalar, T_Params::counterSize>;

    private:
        // only the CRTP engine may create the base
        constexpr explicit PhiloxBaseCommon(State&& state) : state(std::move(state))
        {
        }

        friend T_Impl;

    protected:
        /** Advance the \a counter to the next state
         *
         * Increments the passed-in \a counter by one with a 128-bit carry.
         *
         * @param counter reference to the counter which is to be advanced
         */
        template<typename T, auto N>
        static constexpr void advanceCounter(alpaka::Vec<T, N>& counter)
        {
            ++counter[0];

            /* 128-bit carry */
            if(counter[0] == 0)
            {
                ++counter[1];
                if(counter[1] == 0)
                {
                    ++counter[2];
                    if(counter[2] == 0)
                    {
                        ++counter[3];
                    }
                }
            }
        }

        /** Advance the internal state counter by \a offset N-vectors (N = counter size)
         *
         * Advances the internal value of this->state.counter
         *
         * @param offset number of N-vectors to skip
         */
        constexpr void skip4(uint64_t offset)
        {
            Counter& counter = this->state.counter;
            Counter temp = counter;
            counter[0] += low32Bits(offset);
            counter[1] += high32Bits(offset) + (counter[0] < temp[0] ? 1 : 0);
            counter[2] += (counter[0] < temp[1] ? 1u : 0u);
            counter[3] += (counter[0] < temp[2] ? 1u : 0u);
        }

        /** Advance the counter by the length of \a subsequence
         *
         * Advances the internal value of this->state.counter
         *
         * @param subsequence number of subsequences to skip
         */
        constexpr void skipSubsequence(uint64_t subsequence)
        {
            Counter& counter = this->state.counter;
            Counter temp = counter;
            counter[2] += low32Bits(subsequence);
            counter[3] += high32Bits(subsequence) + (counter[2] < temp[2] ? 1 : 0);
        }
    };
} // namespace alpaka::rand::engine::internal
