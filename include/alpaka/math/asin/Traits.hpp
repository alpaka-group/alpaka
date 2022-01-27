/* Copyright 2022 Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <cmath>

namespace alpaka
{
    namespace math
    {
        struct ConceptMathAsin
        {
        };

        namespace traits
        {
            //! The asin trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Asin
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
                {
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find asin(TArg) in the namespace of your type.
                    using std::asin;
                    return asin(arg);
                }
            };
        } // namespace traits

        //! Computes the principal value of the arc sine.
        //!
        //! The valid real argument range is [-1.0, 1.0]. For other values
        //! the result may depend on the backend and compilation options, will
        //! likely be NaN.
        //!
        //! \tparam TArg The arg type.
        //! \param asin_ctx The object specializing Asin.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto asin(T const& asin_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathAsin, T>;
            return traits::Asin<ImplementationBase, TArg>{}(asin_ctx, arg);
        }
    } // namespace math
} // namespace alpaka
