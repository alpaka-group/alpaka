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

#ifdef ALPAKA_ACC_CPU_BT_OPENACC2_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_CPU_BT_OPENACC2_ENABLED is set, the compiler has to support OpenAcc 2.0 or higher!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuOpenAcc2.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynBoostAlignedAlloc.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/core/MapIdx.hpp>
#include <alpaka/meta/ApplyTuple.hpp>

#include <boost/align.hpp>

#include <openacc.h>

#include <cassert>
#include <stdexcept>
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
        //! The CPU OpenAcc 2.0 accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOpenAcc2 final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuOpenAcc2(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TSize>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuOpenAcc2(ExecCpuOpenAcc2 const & other) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuOpenAcc2(ExecCpuOpenAcc2 && other) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOpenAcc2 const &) -> ExecCpuOpenAcc2 & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOpenAcc2 &&) -> ExecCpuOpenAcc2 & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ExecCpuOpenAcc2() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator()() const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtent(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));
                auto const blockThreadExtent(
                    workdiv::getWorkDiv<Block, Threads>(*this));
                auto const threadElemExtent(
                    workdiv::getWorkDiv<Thread, Elems>(*this));

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes(
                    meta::apply(
                        [&](TArgs const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccCpuOpenAcc2<TDim, TSize>>(
                                        m_kernelFnObj,
                                        blockThreadExtent,
                                        threadElemExtent,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION
                    << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
                // Bind all arguments except the accelerator.
                // TODO: With C++14 we could create a perfectly argument forwarding function object within the constructor.
                auto const boundKernelFnObj(
                    meta::apply(
                        [this](TArgs const & ... args)
                        {
                            return
                                std::bind(
                                    std::ref(m_kernelFnObj),
                                    std::placeholders::_1,
                                    std::ref(args)...);
                        },
                        m_args));

                // The number of blocks in the grid.
                TSize const gridBlockCount(gridBlockExtent.prod());
                // The number of threads in a block.
                TSize const blockThreadCount(blockThreadExtent.prod());

                #pragma acc data copyin(gridBlockExtent), copyin(gridBlockCount), copyin(blockThreadExtent), copyin(blockThreadCount), copyin(blockSharedMemDynSizeBytes)
                {
                    // Execute on the host.
                    // OpenACC | CUDA
                    // -----------------
                    // gang    | block
                    // worker  | warp
                    // vector  | thread
                    #pragma acc parallel num_gangs(gridBlockCount) vector_length(blockThreadCount) if(0)
                    {
                        // Distribute this loop across the grid blocks.
                        #pragma acc loop gang
                        for(TSize b = 0u; b<gridBlockCount; ++b)
                        {
                            // Allocate block shared memory.
                            block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc blockSharedMemDyn(
                                blockSharedMemDynSizeBytes);

                            // Execute the threads in parallel.
                            #pragma acc loop vector
                            for(TSize t = 0u; t<blockThreadCount; ++t)
                            {
                                // We create an accelerator object for each grid block.
                                acc::AccCpuOpenAcc2<TDim, TSize> acc(
                                    *static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this),
                                    block::shared::dyn::getMem<std::uint8_t>(blockSharedMemDyn));

                                const Vec<dim::DimInt<1u>, TSize> gridBlockIdx(b);
                                acc.m_gridBlockIdx = core::mapIdx<TDim::value>(
                                    gridBlockIdx,
                                    gridBlockExtent);

                                const Vec<dim::DimInt<1u>, TSize> blockThreadIdx(t);
                                acc.m_blockThreadIdx = core::mapIdx<TDim::value>(
                                    blockThreadIdx,
                                    blockThreadExtent);

                                boundKernelFnObj(
                                    acc);

                                //FIXME: block::sync::syncBlockThreads(acc);
                            }
                        }
                    }
                }
            }

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOpenAcc2<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenAcc 2.0 executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 executor platform type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenAcc 2.0 executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuOpenAcc2<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
