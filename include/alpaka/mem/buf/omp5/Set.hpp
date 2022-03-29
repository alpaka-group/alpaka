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
}

namespace alpaka::trait
{
    //! The OMP5 device memory set trait specialization.
    template<typename TDim>
    struct CreateTaskMemset<TDim, DevOmp5>
    {
        template<typename TExtent, typename TView>
        ALPAKA_FN_HOST static auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& extent)
        {
            using Idx = typename alpaka::trait::IdxType<TExtent>::type;
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
            std::cout << "Set TDim=" << TDim::value << " pitch=" << pitch << " byteExtent=" << byteExtent << std::endl;
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
        ALPAKA_FN_HOST static auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& /* extent */)
        {
            static_assert(Dim<TView>::value == 0u, "The view is required to have dimensionality 0!");
            static_assert(Dim<TExtent>::value == 0u, "The extent is required to have dimensionality 0!");

            using ExtIdx = Idx<TExtent>;
            using Dim0 = DimInt<0u>;
            using Acc = AccOmp5<Dim0, ExtIdx>;
            auto kernel = [] ALPAKA_FN_ACC(const Acc&, std::uint8_t b, std::uint8_t* mem)
            {
                // for zero dimensional views, we just set the bytes of a single element
                for(auto i = 0u; i < sizeof(Elem<TView>); i++)
                    mem[i] = b;
            };
            using Vec0D = Vec<Dim0, ExtIdx>;
            return createTaskKernel<Acc>(
                WorkDivMembers<Dim0, ExtIdx>{Vec0D{}, Vec0D{}, Vec0D{}},
                kernel,
                byte,
                reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(view)));
        }
    };
} // namespace alpaka::trait

#endif
