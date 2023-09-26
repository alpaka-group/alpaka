/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <tuple>

namespace alpaka::test
{
    //! A std::tuple holding idx types.
    using TestIdxs = std::tuple<
    // size_t is most probably identical to either std::uint64_t or std::uint32_t.
    // This would lead to duplicate tests (especially test names) which is not allowed.
    // std::size_t,
#if !defined(ALPAKA_CI)
        std::int64_t,
#endif
        std::uint64_t,
        std::int32_t
#if !defined(ALPAKA_CI)
        ,
        std::uint32_t
#endif
        // index type must be >=32bit
        >;
} // namespace alpaka::test
