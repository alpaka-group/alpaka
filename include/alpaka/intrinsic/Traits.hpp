/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The intrinsic specifics
    namespace intrinsic
    {
        struct ConceptIntrinsic{};

        //-----------------------------------------------------------------------------
        //! The intrinsics traits.
        namespace traits
        {
            //#############################################################################
            //! The popcount trait.
            template<
                typename TWarp,
                typename TSfinae = void>
            struct Popcount;
        }

        //-----------------------------------------------------------------------------
        //! Returns the number of 1 bits in the given 32-bit value.
        //!
        //! \tparam TIntrinsic The intrinsic implementation type.
        //! \param intrinsic The intrinsic implementation.
        //! \param value The input value.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIntrinsic>
        ALPAKA_FN_ACC auto popcount(
            TIntrinsic const & intrinsic,
            unsigned int value)
        -> int
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
            return traits::Popcount<
                ImplementationBase>
            ::popcount(
                intrinsic,
                value);
        }

        //-----------------------------------------------------------------------------
        //! Returns the number of 1 bits in the given 64-bit value.
        //!
        //! \tparam TIntrinsic The intrinsic implementation type.
        //! \param intrinsic The intrinsic implementation.
        //! \param value The input value.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIntrinsic>
        ALPAKA_FN_ACC auto popcount(
            TIntrinsic const & intrinsic,
            unsigned long long value)
        -> int
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
            return traits::Popcount<
                ImplementationBase>
            ::popcount(
                intrinsic,
                value);
        }
    }
}
