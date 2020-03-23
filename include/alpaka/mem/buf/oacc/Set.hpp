/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

#include <alpaka/dev/DevOacc.hpp>
#include <alpaka/kernel/TaskKernelOacc.hpp>
#include <alpaka/queue/QueueOaccBlocking.hpp>
#include <alpaka/mem/buf/SetKernel.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Utility.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/meta/Integral.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <cstring>

namespace alpaka
{
    namespace dev
    {
        class DevOacc;
    }
}

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The OpenACC device memory set trait specialization.
                template<
                    typename TDim>
                struct CreateTaskSet<
                    TDim,
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto createTaskSet(
                        TView & view,
                        std::uint8_t const & byte,
                        TExtent const & extent)
                    {
                        using Idx = typename idx::traits::IdxType<TExtent>::type;
                        auto pitch = view::getPitchBytesVec(view);
                        auto byteExtent = extent::getExtentVec(extent);
                        byteExtent[TDim::value-1] *= static_cast<Idx>(sizeof(elem::Elem<TView>));
                        constexpr auto lastDim = TDim::value - 1;

                        if(pitch[0] <= 0)
                            return kernel::createTaskKernel<acc::AccOacc<TDim,Idx>>(
                                    workdiv::WorkDivMembers<TDim, Idx>(
                                        vec::Vec<TDim, Idx>::zeros(),
                                        vec::Vec<TDim, Idx>::zeros(),
                                        vec::Vec<TDim, Idx>::zeros()),
                                    view::MemSetKernel(),
                                    byte,
                                    reinterpret_cast<std::uint8_t*>(alpaka::mem::view::getPtrNative(view)),
                                    byteExtent,
                                    pitch
                                ); // NOP if size is zero

                        std::cout << "Set TDim=" << TDim::value << " pitch=" << pitch << " byteExtent=" << byteExtent << std::endl;
                        auto elementsPerThread = vec::Vec<TDim, Idx>::all(static_cast<Idx>(1u));
                        elementsPerThread[lastDim] = 4;
                        // Let alpaka calculate good block and grid sizes given our full problem extent
                        workdiv::WorkDivMembers<TDim, Idx> const workDiv(
                            workdiv::getValidWorkDiv<acc::AccOacc<TDim,Idx>>(
                                dev::getDev(view),
                                byteExtent,
                                elementsPerThread,
                                false,
                                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));
                        return
                            kernel::createTaskKernel<acc::AccOacc<TDim,Idx>>(
                                    workDiv,
                                    view::MemSetKernel(),
                                    byte,
                                    reinterpret_cast<std::uint8_t*>(alpaka::mem::view::getPtrNative(view)),
                                    byteExtent,
                                    pitch
                                );
                    }
                };
            }
        }
    }
}

#endif
