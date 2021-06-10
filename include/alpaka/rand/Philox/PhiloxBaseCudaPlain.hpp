/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/mulhilo.hpp>

#include <utility>

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

namespace alpaka
{
    namespace rand
    {
        namespace engine
        {
            /** Philox backend using plain CUDA uintN types for the storage of Key and Counter
             *
             * @tparam TParams Philox algorithm parameters \sa PhiloxParams
             * @tparam TImpl engine type implementation (CRTP)
             */
            template<typename TParams, typename TImpl>
            class PhiloxBaseCudaPlain
            {
                static_assert(TParams::counterSize == 4, "GPU Philox implemented only for counters of width == 4");

            public:
                using Counter = uint4;
                using Key = uint2;

                static constexpr uint64_t WEYL_64_0 = 0x9E3779B97F4A7C15; /* golden ratio */
                static constexpr uint64_t WEYL_64_1 = 0xBB67AE8584CAA73B; /* sqrt(3)-1 */

                static constexpr uint32_t WEYL_32_0 = hi32(WEYL_64_0);
                static constexpr uint32_t WEYL_32_1 = hi32(WEYL_64_1);

                static constexpr uint32_t MULTIPLITER_4x32_0 = 0xCD9E8D57;
                static constexpr uint32_t MULTIPLITER_4x32_1 = 0xD2511F53;

            protected:
                ALPAKA_FN_HOST_ACC auto singleRound(Counter const& counter, Key const& key)
                {
                    auto [H0, L0] = mulhilo32(counter[0], this->MULTIPLITER_4x32_0);
                    auto [H1, L1] = mulhilo32(counter[2], this->MULTIPLITER_4x32_1);
                    return Counter{H1 ^ counter.x ^ key.x, L1, H0 ^ counter.z ^ key.y, L0};
                }

                ALPAKA_FN_HOST_ACC auto bumpKey(Key const& key)
                {
                    return Key{key.x + WEYL_32_0, key.y + WEYL_32_1};
                }

                ALPAKA_FN_HOST_ACC void advanceCounter(Counter& counter)
                {
                    counter.x++;
                    /* 128-bit carry */
                    if(counter.x == 0)
                    {
                        counter.y++;
                        if(counter.y == 0)
                        {
                            counter.z++;
                            if(counter.z == 0)
                            {
                                counter.w++;
                            }
                        }
                    }
                }

                ALPAKA_FN_HOST_ACC void skip4(uint64_t offset)
                {
                    Counter& counter = static_cast<TImpl*>(this)->state.counter;
                    Counter temp = counter;
                    counter.x += lo32(offset);
                    counter.y += hi32(offset) + (counter.x < temp.x ? 1 : 0);
                    counter.z += (counter.x < temp.y ? 1 : 0);
                    counter.w += (counter.x < temp.z ? 1 : 0);
                }

                ALPAKA_FN_HOST_ACC void skipSubsequence(uint64_t subsequence)
                {
                    Counter& counter = static_cast<TImpl*>(this)->state.counter;
                    Counter temp = counter;
                    counter.z += lo32(subsequence);
                    counter.w += hi32(subsequence) + (counter.z < temp.z ? 1 : 0);
                }
            };
        } // namespace engine
    } // namespace rand
} // namespace alpaka

#endif
