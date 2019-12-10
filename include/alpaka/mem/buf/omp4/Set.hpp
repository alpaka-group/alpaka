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

                    template<
                        typename TDim,
                        typename TIdx>
                    ALPAKA_FN_HOST_ACC auto pitchVecToExtentVec(
                        vec::Vec<TDim, TIdx> pitch)
                    -> vec::Vec<TDim, TIdx>
                    {
                        for(unsigned int i = 0; i < static_cast<unsigned int>(TDim::value) - 1u; ++i)
                        {
                            pitch[i] /= pitch[i+1u];
                        }
                        return pitch;
                    }

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
                            typename TExtent,
                            typename TPitch>
                        ALPAKA_FN_ACC auto operator()(
                            TAcc const & acc,
                            TElem const val,
                            TElem * dst,
                            TExtent extent,
                            TPitch pitch) const
                        -> void
                        {
                            using Idx = typename idx::traits::IdxType<TExtent>::type;
                            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
                            auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
                            auto const idxThreadFirstElem = idx::getIdxThreadFirstElem(acc, gridThreadIdx, threadElemExtent);
                            auto idx = idx::mapIdxPitch<1u, dim::Dim<TAcc>::value>(idxThreadFirstElem, pitch)[0];
                            constexpr auto lastDim = dim::Dim<TAcc>::value - 1;
                            const auto lastIdx = idx +
                                std::min(threadElemExtent[lastDim], static_cast<Idx>(extent[lastDim]-idxThreadFirstElem[lastDim]));

                            if ([&idxThreadFirstElem, &extent](){
                                    for(auto i = 0u; i < dim::Dim<TAcc>::value; ++i)
                                        if(idxThreadFirstElem[i] >= extent[i])
                                            return false;
                                    return true;
                                }())
                            {
                                // assuming elements = {1,1,...,1,n}
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
                            kernel::createTaskKernel<acc::AccCpuOmp4<TDim,typename idx::traits::IdxType<TExtent>::type>>(
                                workdiv::WorkDivMembers<TDim, typename idx::traits::IdxType<TExtent>::type>(
                                    vec::Vec<TDim, typename idx::traits::IdxType<TExtent>::type>::ones(),
                                    vec::Vec<TDim, typename idx::traits::IdxType<TExtent>::type>::ones(),
                                    vec::Vec<TDim, typename idx::traits::IdxType<TExtent>::type>::ones()),
                                view::omp4::detail::MemSetKernel(),
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
                            return kernel::createTaskKernel<acc::AccCpuOmp4<TDim,Idx>>(
                                    workdiv::WorkDivMembers<TDim, Idx>(
                                        vec::Vec<TDim, Idx>::ones(),
                                        vec::Vec<TDim, Idx>::ones(),
                                        vec::Vec<TDim, Idx>::ones()),
                                    view::omp4::detail::MemSetKernel(),
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
                            workdiv::getValidWorkDiv<acc::AccCpuOmp4<TDim,Idx>>(
                                dev::getDev(view),
                                byteExtent,
                                elementsPerThread,
                                false,
                                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));
                        return
                            kernel::createTaskKernel<acc::AccCpuOmp4<TDim,Idx>>(
                                    workDiv,
                                    view::omp4::detail::MemSetKernel(),
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
