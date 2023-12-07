/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/dim/TestDims.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("mapIdx", "[idx]", alpaka::test::TestDims)
{
    using Dim = TestType;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const extentNd = alpaka::test::extentBuf<Dim, Idx>;
    auto const idxNd = extentNd - Vec::all(4u);

    auto const idx1d = alpaka::mapIdx<1u>(idxNd, extentNd);

    auto const idxNdResult = alpaka::mapIdx<Dim::value>(idx1d, extentNd);

    REQUIRE(idxNd == idxNdResult);
}
