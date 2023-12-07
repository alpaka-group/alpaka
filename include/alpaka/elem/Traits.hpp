/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    //! The element trait.
    namespace trait
    {
        //! The element type trait.
        template<typename TView, typename TSfinae = void>
        struct ElemType;
    } // namespace trait

    //! The element type trait alias template to remove the ::type.
    template<typename TView>
    using Elem = std::remove_volatile_t<typename trait::ElemType<TView>::type>;

    // Trait specializations for unsigned integral types.
    namespace trait
    {
        //! The fundamental type elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<std::is_fundamental_v<T>>>
        {
            using type = T;
        };
    } // namespace trait
} // namespace alpaka
