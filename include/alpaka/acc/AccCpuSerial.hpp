/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers
#include <alpaka/idx/gb/IdxGbRef.hpp>           // IdxGbRef
#include <alpaka/idx/bt/IdxBtZero.hpp>          // IdxBtZero
#include <alpaka/atomic/AtomicNoOp.hpp>         // AtomicNoOp
#include <alpaka/math/MathStl.hpp>              // MathStl
#include <alpaka/block/shared/BlockSharedAllocNoSync.hpp>  // BlockSharedAllocNoSync
#include <alpaka/block/sync/BlockSyncNoOp.hpp>  // BlockSyncNoOp

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                // dev::traits::DevType
#include <alpaka/exec/Traits.hpp>               // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>               // size::traits::SizeType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

#include <memory>                               // std::unique_ptr
#include <typeinfo>                             // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuSerial;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU serial accelerator.
        //!
        //! This accelerator allows serial kernel execution on a CPU device.
        //! The block size is restricted to 1x1x1 and all blocks are executed serially so there is no parallelism at all.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class AccCpuSerial final :
            public workdiv::WorkDivMembers<TDim, TSize>,
            public idx::gb::IdxGbRef<TDim, TSize>,
            public idx::bt::IdxBtZero<TDim, TSize>,
            public atomic::AtomicNoOp,
            public math::MathStl,
            public block::shared::BlockSharedAllocNoSync,
            public block::sync::BlockSyncNoOp
        {
        public:
            // Partial specialization with the correct TDim and TSize is not allowed.
            template<
                typename TDim2,
                typename TSize2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::exec::ExecCpuSerial;

        private:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuSerial(
                TWorkDiv const & workDiv) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    idx::gb::IdxGbRef<TDim, TSize>(m_gridBlockIdx),
                    idx::bt::IdxBtZero<TDim, TSize>(),
                    atomic::AtomicNoOp(),
                    math::MathStl(),
                    block::shared::BlockSharedAllocNoSync(),
                    block::sync::BlockSyncNoOp(),
                    m_gridBlockIdx(Vec<TDim, TSize>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            // Do not copy most members because they are initialized by the executor for each execution.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuSerial(AccCpuSerial const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuSerial(AccCpuSerial &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuSerial const &) -> AccCpuSerial & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuSerial &&) -> AccCpuSerial & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AccCpuSerial() = default;

            //-----------------------------------------------------------------------------
            //! \return The pointer to the externally allocated block shared memory.
            //-----------------------------------------------------------------------------
            template<
                typename T>
            ALPAKA_FN_ACC_NO_CUDA auto getBlockSharedExternMem() const
            -> T *
            {
                return reinterpret_cast<T*>(m_externalSharedMem.get());
            }

        private:
            // getIdx
            alignas(16u) Vec<TDim, TSize> mutable m_gridBlockIdx;    //!< The index of the currently executed block.

            // getBlockSharedExternMem
            std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_externalSharedMem;  //!< External block shared memory.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccCpuSerial<TDim, TSize>>
            {
                using type = acc::AccCpuSerial<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU serial accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccCpuSerial<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

                    return {
                        // m_multiProcessorCount
                        static_cast<TSize>(1),
                        // m_blockThreadsCountMax
                        static_cast<TSize>(1),
                        // m_blockThreadExtentsMax
                        Vec<TDim, TSize>::ones(),
                        // m_gridBlockExtentsMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max())};
                }
            };
            //#############################################################################
            //! The CPU serial accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccCpuSerial<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuSerial<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccCpuSerial<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU serial accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                acc::AccCpuSerial<TDim, TSize>>
            {
                using type = dev::DevManCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccCpuSerial<TDim, TSize>>
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
            //! The CPU serial accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuSerial<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuSerial<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccCpuSerial<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
