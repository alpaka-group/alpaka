/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/PhiloxBaseArrayLike.hpp>
#include <alpaka/rand/Philox/mulhilo.hpp>

//#include "alpaka_rand/rand/detail/PhiloxBaseArrayLike.hpp"

#include <utility>


namespace alpaka
{
    namespace rand
    {
        namespace engine
        {
            /** Philox backend using std::array for Key and Counter storage
             *
             * @tparam TParams Philox algorithm parameters \sa PhiloxParams
             * @tparam TImpl engine type implementation (CRTP)
             */
            template<typename TParams, typename TImpl>
            class PhiloxBaseStdArray
                : public PhiloxBaseArrayLike<
                      TParams,
                      std::array<uint32_t, TParams::counterSize>, // Counter
                      std::array<uint32_t, TParams::counterSize / 2>, // Key
                      TImpl>
            {
            public:
                using Counter = std::array<uint32_t, TParams::counterSize>; ///< Counter type = std::array
                using Key = std::array<uint32_t, TParams::counterSize / 2>; ///< Key type = std::array
            };
        } // namespace engine
    } // namespace rand
} // namespace alpaka
