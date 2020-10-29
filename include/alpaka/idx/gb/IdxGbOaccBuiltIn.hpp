/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

#include <alpaka/idx/Traits.hpp>
#include <alpaka/idx/gb/IdxGbLinear.hpp>

namespace alpaka
{
    namespace gb
    {
        //#############################################################################
        //! The OpenACC ND index provider dummy.
        template<
            typename TDim,
            typename TIdx>
        class IdxGbOaccBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            IdxGbOaccBuiltIn() = default;
            //-----------------------------------------------------------------------------
            IdxGbOaccBuiltIn(IdxGbOaccBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            IdxGbOaccBuiltIn(IdxGbOaccBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbOaccBuiltIn const & ) -> IdxGbOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbOaccBuiltIn &&) -> IdxGbOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~IdxGbOaccBuiltIn() = default;

            using BlockShared = IdxGbLinear<TDim, TIdx>;
        };
    }
}

#endif
