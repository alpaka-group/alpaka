/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/dev/DevOmp5.hpp>
#include <alpaka/kernel/TaskKernelOmp5.hpp>
#include <alpaka/queue/QueueOmp5Blocking.hpp>
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
        class DevOmp5;
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
                //! The OMP5 device memory set trait specialization.
                template<
                    typename TDim>
                struct CreateTaskSet<
                    TDim,
                    dev::DevOmp5>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto createTaskSet(
                        TView & view,
                        std::uint8_t const & byte,
                        TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> decltype(
                            kernel::createTaskKernel<acc::AccOmp5<TDim,typename idx::traits::IdxType<TExtent>::type>>(
                                workdiv::WorkDivMembers<TDim, typename idx::traits::IdxType<TExtent>::type>(
                                    vec::Vec<TDim, typename idx::traits::IdxType<TExtent>::type>::ones(),
                                    vec::Vec<TDim, typename idx::traits::IdxType<TExtent>::type>::ones(),
                                    vec::Vec<TDim, typename idx::traits::IdxType<TExtent>::type>::ones()),
                                view::MemSetKernel(),
                                byte,
                                reinterpret_cast<std::uint8_t*>(alpaka::mem::view::getPtrNative(view)),
                                alpaka::core::declval<decltype(extent::getExtentVec(extent))&>(),
                                alpaka::core::declval<decltype(view::getPitchBytesVec(view))&>()
                            )
                        )
#endif
                    {
                        using Idx = typename idx::traits::IdxType<TExtent>::type;
                        auto pitch = view::getPitchBytesVec(view);
                        auto byteExtent = extent::getExtentVec(extent);
                        byteExtent[TDim::value-1] *= static_cast<Idx>(sizeof(elem::Elem<TView>));
                        constexpr auto lastDim = TDim::value - 1;

                        if(pitch[0] <= 0)
                            return kernel::createTaskKernel<acc::AccOmp5<TDim,Idx>>(
                                    workdiv::WorkDivMembers<TDim, Idx>(
                                        vec::Vec<TDim, Idx>::ones(),
                                        vec::Vec<TDim, Idx>::ones(),
                                        vec::Vec<TDim, Idx>::ones()),
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
                            workdiv::getValidWorkDiv<acc::AccOmp5<TDim,Idx>>(
                                dev::getDev(view),
                                byteExtent,
                                elementsPerThread,
                                false,
                                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));
                        return
                            kernel::createTaskKernel<acc::AccOmp5<TDim,Idx>>(
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
