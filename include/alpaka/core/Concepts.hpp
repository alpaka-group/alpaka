/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    namespace concepts
    {
        //#############################################################################
        //! Tag used in class inheritance hierarchies that describes that a specific concept (TConcept)
        //! is implemented by the given base class (TBase).
        template<
            typename TConcept,
            typename TBase>
        struct Implements
        {
        };

        namespace detail
        {
            //#############################################################################
            //! Given a derived class (TDerived) it finds the base class (TBase) which implements the given concept (TConcept).
            template<
                typename TConcept,
                typename TDerived>
            struct ImplementationBaseType
            {
                template <typename TBase>
                static auto base(Implements<TConcept, TBase>*) -> TBase;

                using type = decltype(base(std::declval<TDerived*>()));

                static_assert(std::is_base_of<type, TDerived>::value, "The type implementing the concept has to be a publicly accessible base class!");
            };
        }

        //#############################################################################
        //! Given a derived class (TDerived) it finds the base class (TBase) which implements the given concept (TConcept).
        template<
            typename TConcept,
            typename TDerived>
        using ImplementationBase = typename detail::ImplementationBaseType<TConcept, TDerived>::type;
    }
}
