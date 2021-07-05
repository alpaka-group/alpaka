/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/PhiloxConstants.hpp>
#include <alpaka/rand/Philox/mulhilo.hpp>

#include <utility>

namespace alpaka
{
    namespace rand
    {
        namespace engine
        {
            /** Base class for Counter and Key types with array-like interface
             *
             * Provides the implementation of key bumping, counter advancing, counter skipping, and subsequence
             * skipping for types that support subscripting by operator [].
             *
             * @tparam TCounter Counter type
             * @tparam TKey Key type
             * @tparam TParams Philox algorithm parameters \sa PhiloxParams
             * @tparam TImpl engine type implementation (CRTP)
             */
            template<typename TParams, typename TCounter, typename TKey, typename TImpl>
            class PhiloxBaseArrayLike : public PhiloxConstants<TParams>
            {
            protected:
                using Counter = TCounter;
                using Key = TKey;

                /** Single round of the Philox shuffle
                 *
                 * @param counter state of the counter
                 * @param key value of the key
                 * @return shuffled counter
                 */
                ALPAKA_FN_HOST_ACC auto singleRound(Counter const& counter, Key const& key)
                {
                    //TODO:
                    auto [H0, L0] = mulhilo32(counter[0], this->MULTIPLITER_4x32_0);
                    auto [H1, L1] = mulhilo32(counter[2], this->MULTIPLITER_4x32_1);
                    return Counter{H1 ^ counter[1] ^ key[0], L1, H0 ^ counter[3] ^ key[1], L0};
                }

                /** Bump the \a key by the Weyl sequence step parameter
                 *
                 * @param key the key to be bumped
                 * @return the bumped key
                 */
                ALPAKA_FN_HOST_ACC auto bumpKey(Key const& key)
                {
                    return Key{key[0] + this->WEYL_32_0, key[1] + this->WEYL_32_1};
                }

                /** Advance the \a counter to the next state
                 *
                 * Increments the passed-in \a counter by one with a 128-bit carry.
                 *
                 * @param counter reference to the counter which is to be advanced
                 */
                ALPAKA_FN_HOST_ACC void advanceCounter(Counter& counter)
                {
                    counter[0]++;
                    /* 128-bit carry */
                    if(counter[0] == 0)
                    {
                        counter[1]++;
                        if(counter[1] == 0)
                        {
                            counter[2]++;
                            if(counter[2] == 0)
                            {
                                counter[3]++;
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
                ALPAKA_FN_HOST_ACC void skip4(uint64_t offset)
                {
                    Counter& counter = static_cast<TImpl*>(this)->state.counter;
                    Counter temp = counter;
                    counter[0] += lo32(offset);
                    counter[1] += hi32(offset) + (counter[0] < temp[0] ? 1 : 0);
                    counter[2] += (counter[0] < temp[1] ? 1 : 0);
                    counter[3] += (counter[0] < temp[2] ? 1 : 0);
                }

                /** Advance the counter by the length of \a subsequence
                 *
                 * Advances the internal value of this->state.counter
                 *
                 * @param subsequence number of subsequences to skip
                 */
                ALPAKA_FN_HOST_ACC void skipSubsequence(uint64_t subsequence)
                {
                    Counter& counter = static_cast<TImpl*>(this)->state.counter;
                    Counter temp = counter;
                    counter[2] += lo32(subsequence);
                    counter[3] += hi32(subsequence) + (counter[2] < temp[2] ? 1 : 0);
                }
            };
        } // namespace engine
    } // namespace rand
} // namespace alpaka
