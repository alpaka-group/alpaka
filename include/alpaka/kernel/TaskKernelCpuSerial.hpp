/**
* \file
* Copyright 2014-2019 Benjamin Worpitz, Rene Widera
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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace kernel
    {
        //#############################################################################
        //! The CPU serial execution task implementation.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelCpuSerial final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelCpuSerial(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelCpuSerial(TaskKernelCpuSerial const &) = default;
            //-----------------------------------------------------------------------------
            TaskKernelCpuSerial(TaskKernelCpuSerial &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuSerial const &) -> TaskKernelCpuSerial & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuSerial &&) -> TaskKernelCpuSerial & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelCpuSerial() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
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
                                    acc::AccCpuSerial<TDim, TIdx>>(
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

                acc::AccCpuSerial<TDim, TIdx> acc(
                    *static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this),
                    blockSharedMemDynSizeBytes);

                if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error("A block for the serial accelerator can only ever have one single thread!");
                }

                // Execute the blocks serially.
                meta::ndLoopIncIdx(
                    gridBlockExtent,
                    [&](vec::Vec<TDim, TIdx> const & blockThreadIdx)
                    {
                        acc.m_gridBlockIdx = blockThreadIdx;

                        boundKernelFnObj(
                            acc);

                        // After a block has been processed, the shared memory has to be deleted.
                        block::shared::st::freeMem(acc);
                    });
            }

        private:
            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuSerial<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU serial execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU serial execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU serial execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
