/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/core/Utility.hpp>
#    include <alpaka/dev/DevOmp5.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/idx/Accessors.hpp>
#    include <alpaka/kernel/TaskKernelOmp5.hpp>
#    include <alpaka/mem/buf/SetKernel.hpp>
#    include <alpaka/mem/view/Traits.hpp>
#    include <alpaka/meta/Integral.hpp>
#    include <alpaka/queue/QueueOmp5Blocking.hpp>
#    include <alpaka/vec/Vec.hpp>
#    include <alpaka/workdiv/WorkDivHelpers.hpp>

#    include <cstring>

namespace alpaka
{
    class DevOmp5;

    namespace traits
    {
        //! The OMP5 device memory set trait specialization.
        template<typename TDim>
        struct CreateTaskMemset<TDim, DevOmp5>
        {
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& extent)
            {
                using Idx = typename alpaka::traits::IdxType<TExtent>::type;
                auto pitch = getPitchBytesVec(view);
                auto byteExtent = getExtentVec(extent);
                constexpr auto lastDim = TDim::value - 1;
                byteExtent[lastDim] *= static_cast<Idx>(sizeof(Elem<TView>));

                if(pitch[0] == 0)
                {
                    return createTaskKernel<AccOmp5<TDim, Idx>>(
                        WorkDivMembers<TDim, Idx>(
                            Vec<TDim, Idx>::zeros(),
                            Vec<TDim, Idx>::zeros(),
                            Vec<TDim, Idx>::zeros()),
                        MemSetKernel(),
                        byte,
                        reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(view)),
                        byteExtent,
                        pitch); // NOP if size is zero
                }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << "Set TDim=" << TDim::value << " pitch=" << pitch << " byteExtent=" << byteExtent
                          << std::endl;
#    endif
                auto elementsPerThread = Vec<TDim, Idx>::all(static_cast<Idx>(1u));
                elementsPerThread[lastDim] = 4;
                // Let alpaka calculate good block and grid sizes given our full problem extent
                WorkDivMembers<TDim, Idx> const workDiv(getValidWorkDiv<AccOmp5<TDim, Idx>>(
                    getDev(view),
                    byteExtent,
                    elementsPerThread,
                    false,
                    alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));
                return createTaskKernel<AccOmp5<TDim, Idx>>(
                    workDiv,
                    MemSetKernel(),
                    byte,
                    reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(view)),
                    byteExtent,
                    pitch);
            }
        };
        //! The OMP5 device scalar memory set trait specialization.
        template<>
        struct CreateTaskMemset<DimInt<0u>, DevOmp5>
        {
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(
                TView& view,
                std::uint8_t const& byte,
                TExtent const& /* extent */)
            {
                using Idx = typename alpaka::traits::IdxType<TExtent>::type;
                using Dim1D = DimInt<1u>;
                using Vec1D = Vec<Dim1D, Idx>;

                static_assert(Dim<TView>::value == 0u, "The view is required to have dimensionality 0!");
                static_assert(Dim<TExtent>::value == 0u, "The extent is required to have dimensionality 0!");

                return createTaskKernel<AccOmp5<Dim1D, Idx>>(
                    WorkDivMembers<Dim1D, Idx>(Vec1D::zeros(), Vec1D::zeros(), Vec1D::zeros()),
                    MemSetKernel(),
                    byte,
                    reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(view)),
                    Vec1D(sizeof(Elem<TView>)),
                    Vec1D::zeros());
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
