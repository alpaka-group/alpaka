/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <random>

/**
 * @namespace test
 * @brief Only contains fillWithRndArgs.
 * @fn fillWithRndArgs
 * @tparam Data The used Buffer-type.
 * @param buffer The buffer that should be filled.
 * @param size The size of the used buffer.
 * @param range The Range, around Zero, for the data.
 */

namespace test
{
    template< typename Data >
    auto fillWithRndArgs(
        Data * buffer,
        size_t size,
        size_t range
    ) -> unsigned long
    {
        std::random_device rd {};
        unsigned long seed = rd( );
        // for every accelerator type one seed will be generated
        std::default_random_engine eng { seed };
        // these pseudo-random numbers are implementation specific
        // and are not portable
        std::uniform_real_distribution< Data > dist(
            0,
            range
        );

        // Initiate the arguments.
        for( size_t i( 0 ); i < size / 2 - 1; ++i )
        {
            buffer[i] = dist( eng );

        }
        // Define the middle of the args-buffer as zeros
        buffer[size / 2 - 1] = 0.0;

        buffer[size / 2] = -0.0;

        // Change the Range for the random arguments to [-randomRange, 0]
        for( size_t i( size / 2 + 1 ); i < size; ++i )
        {
            buffer[i] = dist( eng ) - range;
        }

        return seed;

    }
}