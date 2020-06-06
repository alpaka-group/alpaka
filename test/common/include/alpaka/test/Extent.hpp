/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        template<
            typename TIdx>
        struct CreateVecWithIdx
        {
            //#############################################################################
            //! 1D: (16)
            //! 2D: (16, 14)
            //! 3D: (16, 14, 12)
            //! 4D: (16, 14, 12, 10)
            template<
                std::size_t Tidx>
            struct ForExtentBuf
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST_ACC static auto create()
                {
                    return static_cast<TIdx>(16u - (Tidx*2u));
                }
            };

            //#############################################################################
            //! 1D: (11)
            //! 2D: (11, 8)
            //! 3D: (11, 8, 5)
            //! 4D: (11, 8, 5, 2)
            template<
                std::size_t Tidx>
            struct ForExtentSubView
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST_ACC static auto create()
                {
                    return static_cast<TIdx>(11u - (Tidx*3u));
                }
            };

            //#############################################################################
            //! 1D: (2)
            //! 2D: (2, 3)
            //! 3D: (2, 3, 4)
            //! 4D: (2, 3, 4, 5)
            template<
                std::size_t Tidx>
            struct ForOffset
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST_ACC static auto create()
                {
                    return static_cast<TIdx>(2u + Tidx);
                }
            };
        };
    }
}
