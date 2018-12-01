/**
* \file
* Copyright 2018 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_COROUTINES_ENABLED

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuCoroutines.hpp>
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <experimental/coroutine>

#include <future>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace exec
    {
        //#############################################################################
        //! The CPU fibers accelerator executor.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuCoroutines final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuCoroutines(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_boundKernelFnObj(
                        [=](const acc::AccCpuCoroutines<TDim, TIdx>& acc) -> coreturn
                        {
                            co_return kernelFnObj(acc, args...);
                        }),
                    m_blockSharedMemDynSizeBytes(
                        calculateBlockSharedMemDynSizeBytes(
                            workDiv,
                            kernelFnObj,
                            args...))
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION
                    << " blockSharedMemDynSizeBytes: " << m_blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
            }
            //-----------------------------------------------------------------------------
            ExecCpuCoroutines(ExecCpuCoroutines const &) = default;
            //-----------------------------------------------------------------------------
            ExecCpuCoroutines(ExecCpuCoroutines &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuCoroutines const &) -> ExecCpuCoroutines & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuCoroutines &&) -> ExecCpuCoroutines & = default;
            //-----------------------------------------------------------------------------
            ~ExecCpuCoroutines() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()() const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtent(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));

                acc::AccCpuCoroutines<TDim, TIdx> acc(
                    *static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this),
                    m_blockSharedMemDynSizeBytes);

                // Execute the blocks serially.
                meta::ndLoopIncIdx(
                    gridBlockExtent,
                    [&](vec::Vec<TDim, TIdx> const & blockThreadIdx)
                    {
                        acc.m_gridBlockIdx = blockThreadIdx;

                        m_boundKernelFnObj(
                            acc).wait();

                        // After a block has been processed, the shared memory has to be deleted.
                        block::shared::st::freeMem(acc);
                    });
            }

        private:
            template<
                typename TWorkDiv>
            auto calculateBlockSharedMemDynSizeBytes(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) const
            -> TIdx
            {
                auto const blockThreadExtent(
                    workdiv::getWorkDiv<Block, Threads>(workDiv));
                auto const threadElemExtent(
                    workdiv::getWorkDiv<Thread, Elems>(workDiv));

                if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error("A block for the coroutines accelerator can only ever have one single thread!");
                }

                return
                    kernel::getBlockSharedMemDynSizeBytes<
                        acc::AccCpuCoroutines<TDim, TIdx>>(
                            kernelFnObj,
                            blockThreadExtent,
                            threadElemExtent,
                            args...);
            }

            std::function<coreturn(const acc::AccCpuCoroutines<TDim, TIdx>&)> const m_boundKernelFnObj;
            TIdx const m_blockSharedMemDynSizeBytes;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuCoroutines<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuCoroutines<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuCoroutines<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuCoroutines<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuCoroutines<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                exec::ExecCpuCoroutines<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
