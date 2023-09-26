/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/meta/Filter.hpp"
#include "alpaka/meta/NonZero.hpp"

#include <tuple>

namespace alpaka::test
{
    //! A std::tuple holding dimensions.
    using TestDims = std::tuple<
        DimInt<0u>,
        DimInt<1u>,
        DimInt<2u>,
        DimInt<3u>
    // CUDA, HIP and SYCL accelerators do not support 4D buffers and 4D acceleration.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
        ,
        DimInt<4u>
#endif
        >;

    //! A std::tuple holding non-zero dimensions.
    //!
    //! NonZeroTestDims = std::tuple<Dim1, Dim2, ... DimN>
    using NonZeroTestDims = meta::Filter<TestDims, meta::NonZero>;

} // namespace alpaka::test
