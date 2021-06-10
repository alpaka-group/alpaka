/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/PhiloxBaseTraits.hpp>
#include <alpaka/rand/Philox/mulhilo.hpp>

#include <utility>

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <alpaka/rand/Philox/helpers/cuintArray.hpp>
#endif


namespace alpaka
{
    namespace rand
    {
        namespace engine
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

                Counter counter; ///< Counter array
                Key key; ///< Key array
            };

            /** Philox engine generating a vector of numbers
             *
             * This engine's operator() will return a vector of numbers corresponding to the full size of its counter.
             * This is a convenience vs. memory size tradeoff since the user has to deal with the output array
             * themselves, but the internal state comprises only of a single counter and a key.
             *
             * @tparam TAcc Accelerator type as defined in alpaka/acc
             * @tparam TParams Basic parameters for the Philox algorithm
             */
            template<typename TAcc, typename TParams>
            class PhiloxVector : public traits::PhiloxBaseTraits<TAcc, TParams, PhiloxVector<TAcc, TParams>>::Base
            {
            public:
                using Trait =
                    typename traits::PhiloxBaseTraits<TAcc, TParams, PhiloxVector<TAcc, TParams>>; ///< Specialization
                                                                                                   ///< for different
                                                                                                   ///< backends
                using Counter = typename Trait::Counter; ///< Backend-dependent Counter type
                using Key = typename Trait::Key; ///< Backend-dependent Key type
                using State = PhiloxStateVector<Counter, Key>; ///< Backend-dependent State type

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
                // XXX: Skips the whole vector! Is it enough to document this behavior?
                ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
                {
                    this->skip4(offset);
                }

            public:
                /** Construct a new Philox engine with vector output
                 *
                 * @param seed Set the Philox generator key
                 * @param subsequence Select a subsequence of size 2^64
                 * @param offset Skip \a offset numbers form the start of the subsequence
                 */
                ALPAKA_FN_HOST_ACC PhiloxVector(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
                    : state{{0, 0, 0, 0}, {static_cast<uint32_t>(lo32(seed)), static_cast<uint32_t>(hi32(seed))}}
                {
                    this->skipSubsequence(subsequence);
                    skip(offset);
                    nextVector();
                }

                ALPAKA_FN_HOST_ACC PhiloxVector(PhiloxVector const& other) : state{other.state}
                {
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
        } // namespace engine
    } // namespace rand
} // namespace alpaka
