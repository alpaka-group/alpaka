/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"

#include <cstddef>

namespace alpaka::test
{
    template<typename TIdx>
    struct CreateVecWithIdx
    {
        //! 1D: (11)
        //! 2D: (11, 10)
        //! 3D: (11, 10, 9)
        //! 4D: (11, 10, 9, 8)
        template<std::size_t Tidx>
        struct ForExtentBuf
        {
            ALPAKA_FN_HOST_ACC static auto create()
            {
                return static_cast<TIdx>(11u - Tidx);
            }
        };

        //! 1D: (8)
        //! 2D: (8, 6)
        //! 3D: (8, 6, 4)
        //! 4D: (8, 6, 4, 2)
        template<std::size_t Tidx>
        struct ForExtentSubView
        {
            ALPAKA_FN_HOST_ACC static auto create()
            {
                return static_cast<TIdx>(8u - (Tidx * 2u));
            }
        };

        //! 1D: (2)
        //! 2D: (2, 3)
        //! 3D: (2, 3, 4)
        //! 4D: (2, 3, 4, 5)
        template<std::size_t Tidx>
        struct ForOffset
        {
            ALPAKA_FN_HOST_ACC static auto create()
            {
                return static_cast<TIdx>(2u + Tidx);
            }
        };
    };
} // namespace alpaka::test
