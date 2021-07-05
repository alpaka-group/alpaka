/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/meta/RepeatN.hpp>
#include <alpaka/rand/Philox/mulhilo.hpp>

#include <utility>


namespace alpaka
{
    namespace rand
    {
        namespace engine
        {
            /** Philox algorithm parameters
             *
             * @tparam TCounterSize number of elements in the counter
             * @tparam TWidth width of one counter element (in bits)
             * @tparam TRounds number of S-box rounds
             */
            // TODO: Not sure where to put this
            template<unsigned TCounterSize, unsigned TWidth, unsigned TRounds>
            struct PhiloxParams
            {
                static unsigned constexpr counterSize = TCounterSize;
                static unsigned constexpr width = TWidth;
                static unsigned constexpr rounds = TRounds;
            };

            /** Common class for Philox family engines
             *
             * Checks the validity of passed-in parameters and calls the \a TBackend methods to perform N rounds of the
             * Philox shuffle.
             *
             * @tparam TBackend device-dependent backend, specifies the array types
             * @tparam TParams Philox algorithm parameters \sa PhiloxParams
             * @tparam TImpl engine type implementation (CRTP)
             */
            template<typename TBackend, typename TParams, typename TImpl>
            class PhiloxBaseCommon : public TBackend
            {
                static unsigned const numRounds = TParams::rounds;
                static unsigned const vectorSize = TParams::counterSize;
                static unsigned const numberWidth = TParams::width;

                static_assert(numRounds > 0, "Number of Philox rounds must be > 0.");
                static_assert(vectorSize % 2 == 0, "Philox counter size must be an even number.");
                static_assert(vectorSize <= 16, "Philox SP network is not specified for sizes > 16.");
                static_assert(numberWidth % 8 == 0, "Philox number width in bits must be a multiple of 8.");

                // static_assert(TWidth == 32 || TWidth == 64, "Philox implemented only for 32 and 64 bit numbers.");
                static_assert(numberWidth == 32, "Philox implemented only for 32 bit numbers.");

            public:
                using Counter = typename TBackend::Counter;
                using Key = typename TBackend::Key;

            protected:
                /** Performs N rounds of the Philox shuffle
                 *
                 * @param counter_in initial state of the counter
                 * @param key_in initial state of the key
                 * @return result of the PRNG shuffle; has the same size as the counter
                 */
                ALPAKA_FN_HOST_ACC auto nRounds(Counter const& counter_in, Key const& key_in) -> Counter
                {
                    Key key{key_in};
                    Counter counter = TBackend::singleRound(counter_in, key);

                    // TODO: Possible premature optimization. Check real performance.
                    meta::RepeatN<numRounds>()(
                        [&]()
                        {
                            key = TBackend::bumpKey(key);
                            counter = TBackend::singleRound(counter, key);
                        });
                    // TODO: Should the key be returned as well??? i.e. should the original key be bumped?

                    return counter;
                }
            };
        } // namespace engine
    } // namespace rand
} // namespace alpaka
