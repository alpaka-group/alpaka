/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/dev/DevOmp4.hpp>
#include <alpaka/kernel/TaskKernelCpuOmp4.hpp>
#include <alpaka/queue/QueueOmp4Blocking.hpp>

#include <alpaka/core/Assert.hpp>
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
        class DevOmp4;
    }
}

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace omp4
            {
                namespace detail
                {
                    //#############################################################################
                    //! The OMP4 device ND memory set kernel.
                    class MemSetKernel
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        //! The kernel entry point.
                        //!
                        //! \tparam TAcc The accelerator environment to be executed on.
                        //! \tparam TElem element type.
                        //! \tparam TExtent extent type.
                        //! \param acc The accelerator to be executed on.
                        //! \param val value to set.
                        //! \param dst target mem ptr.
                        //! \param extent area to set.
                        ALPAKA_NO_HOST_ACC_WARNING
                        template<
                            typename TAcc,
                            typename TElem,
                            typename TExtent>
                        ALPAKA_FN_ACC auto operator()(
                            TAcc const & acc,
                            TElem const val,
                            TElem * dst,
                            TExtent extent) const
                        -> void
                        {
                            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
                            auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
                            auto const idxThreadFirstElem = idx::getIdxThreadFirstElem(acc, gridThreadIdx, threadElemExtent);
                            auto idx = idxThreadFirstElem.prod();
                            constexpr auto lastDim = dim::Dim<TAcc>::value - 1;
                            const auto lastIdx = std::min(idx + threadElemExtent[lastDim], extent[lastDim]);

                            if(idx < extent.prod())
                            {
                                // assuming elements = {1,1,1,...,n}
                                for(; idx<lastIdx; ++idx)
                                {
                                    dst[idx] = val;
                                }
                            }
                        }
                    };
                }
            }

            namespace traits
            {
                //#############################################################################
                //! The OMP4 device memory set trait specialization.
                template<
                    typename TDim>
                struct CreateTaskSet<
                    TDim,
                    dev::DevOmp4>
                {
                    using Idx = std::size_t;
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto createTaskSet(
                        TView & view,
                        std::uint8_t const & byte,
                        TExtent const & extent)
                    -> kernel::TaskKernelCpuOmp4<
                        TDim,
                        Idx,
                        view::omp4::detail::MemSetKernel,
                        std::uint8_t,
                        std::uint8_t*,
                        decltype(alpaka::extent::getExtentVec(extent))
                        >
                    {
                        auto elementsPerThread = vec::Vec<TDim, Idx>::all(1u);
                        elementsPerThread[TDim::value-1] = 4;
                        // Let alpaka calculate good block and grid sizes given our full problem extent
                        alpaka::workdiv::WorkDivMembers<TDim, Idx> const workDiv(
                            alpaka::workdiv::getValidWorkDiv<acc::AccCpuOmp4<TDim,Idx>>(
                                dev::getDev(view),
                                extent,
                                elementsPerThread,
                                false,
                                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));
                        return
                            kernel::createTaskKernel<acc::AccCpuOmp4<TDim,Idx>>(
                                    workDiv,
                                    view::omp4::detail::MemSetKernel(),
                                    byte,
                                    reinterpret_cast<std::uint8_t*>(alpaka::mem::view::getPtrNative(view)),
                                    alpaka::extent::getExtentVec(extent)
                                    );
                    }
                };
            }
        }
    }
}

#endif
