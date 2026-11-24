/* Copyright 2022 Jiri Vyskocil, Rene Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/engine/philox/PhiloxBaseCommon.hpp"
#include "alpaka/rand/engine/philox/multiplyAndSplit64to32.hpp"

#include <utility>

namespace alpaka::rand::engine::internal
{


    /** Philox engine generating a single number
     *
     * This engine's operator() will return a single number. Since the result is the same size as the counter,
     * and so it contains more than one number, it has to be stored between individual invocations of
     * operator(). Additionally a pointer has to be stored indicating which part of the result array is to be
     * returned next.
     *
     * @tparam T_Params Basic parameters for the Philox algorithm
     */
    template<typename T_Params>
    class PhiloxSingle : public PhiloxBaseCommon<T_Params, PhiloxSingle<T_Params>>
    {
    public:
        using Base = PhiloxBaseCommon<T_Params, PhiloxSingle<T_Params>>;

        /// Counter type
        using Counter = typename Base::Counter;
        /// Key type
        using Key = typename Base::Key;
        using State = PhiloxState<Counter, Key, PhiloxSingle<T_Params>>;


    protected:
        /** Advance internal counter to the next value
         *
         * Advances the full internal counter array, resets the position pointer and stores the intermediate
         * result to be recalled when the user requests a number.
         */
        constexpr void advanceState()
        {
            this->advanceCounter(this->state.counter);
            this->state.result = this->nRounds(this->state.counter, this->state.key);
            this->state.position = 0;
        }

        /** Get the next random number and advance internal state
         *
         * The intermediate result stores N = TParams::counterSize numbers. Check if we've already given out
         * all of them. If so, generate a new intermediate result (this also resets the pointer to the position
         * of the actual number). Finally, we return the actual number.
         *
         * @return The next random number
         */
        constexpr auto nextNumber()
        {
            // Element zero will always contain the next valid random number.
            auto result = this->state.result[0];
            ++this->state.position;
            if(this->state.position == T_Params::counterSize)
            {
                advanceState();
            }
            else
            {
                /* Shift state results to allow hard coded access to element zero.
                 * This will avoid high register usage on NVIDIA devices.
                 * @todo Check if this shifting of the result vector is decreasing CPU performance.
                 *       If so this optimization for GPUs (mostly NVIDIA/AMD) should be made optional.
                 */
                this->state.result[0] = this->state.result[1];
                this->state.result[1] = this->state.result[2];
                this->state.result[2] = this->state.result[3];
            }

            return result;
        }

        /// Skips the next \a count numbers of the current block.
        /// Requires state.position + count <= T_Params::counterSize.
        constexpr void skipInBlock(std::uint32_t count)
        {
            this->state.position += count;
            for(; count > 0; --count)
            {
                // Element zero always contains the next number, see nextNumber().
                this->state.result[0] = this->state.result[1];
                this->state.result[1] = this->state.result[2];
                this->state.result[2] = this->state.result[3];
            }
        }

        /// Skips the next \a offset numbers
        constexpr void skip(uint64_t offset)
        {
            static_assert(T_Params::counterSize == 4, "Only counterSize 4 is supported.");
            // Position of the target number in the current block plus the remainder; less than 8.
            uint64_t const inBlock = this->state.position + offset % T_Params::counterSize;
            uint64_t const blocks = offset / T_Params::counterSize + inBlock / T_Params::counterSize;
            uint64_t shifts = offset;
            if(blocks > 0)
            {
                // advanceState() moves from the current block to the next one.
                this->skip4(blocks - 1);
                advanceState();
                shifts = inBlock % T_Params::counterSize;
            }
            skipInBlock(static_cast<std::uint32_t>(shifts));
        }

    public:
        /** Construct a new Philox engine with single-value output
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset numbers from the start of the subsequence
         */
        constexpr PhiloxSingle(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
            : Base(State{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}, {0, 0, 0, 0}, 0u})
        {
            this->skipSubsequence(subsequence);
            this->skip4(offset / T_Params::counterSize);
            advanceState();
            skipInBlock(static_cast<std::uint32_t>(offset % T_Params::counterSize));
        }

        /** Get the next random number
         *
         * @return The next random number
         */
        constexpr auto operator()()
        {
            return nextNumber();
        }
    };
} // namespace alpaka::rand::engine::internal
