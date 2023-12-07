/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    //! Properties to define queue behavior
    namespace property
    {
        //! The caller is waiting until the enqueued task is finished
        struct Blocking;

        //! The caller is NOT waiting until the enqueued task is finished
        struct NonBlocking;
    } // namespace property

    using namespace property;
} // namespace alpaka
