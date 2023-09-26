/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    //! The dimension trait.
    namespace trait
    {
        //! The dimension getter type trait.
        template<typename T, typename TSfinae = void>
        struct DimType;
    } // namespace trait

    //! The dimension type trait alias template to remove the ::type.
    template<typename T>
    using Dim = typename trait::DimType<T>::type;
} // namespace alpaka
