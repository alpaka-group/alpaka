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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/dev/Traits.hpp>                // DevType
#include <alpaka/exec/Traits.hpp>               // ExecType
#include <alpaka/size/Traits.hpp>               // size::SizeT

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // DevCpu

#include <alpaka/core/OpenMp.hpp>

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

#include <memory>                               // std::unique_ptr
#include <typeinfo>                             // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize>
        class ExecCpuOmp2Blocks;

        namespace omp
        {
            namespace omp2
            {
                namespace blocks
                {
                    namespace detail
                    {
                        template<
                            typename TDim,
                            typename TSize>
                        class ExecCpuOmp2BlocksImpl;
                    }
                }
            }
        }
    }
    namespace acc
    {
        //-----------------------------------------------------------------------------
        //! The OpenMP accelerators.
        //-----------------------------------------------------------------------------
        namespace omp
        {
            //-----------------------------------------------------------------------------
            //! The CPU OpenMP 2.0 block accelerator.
            //-----------------------------------------------------------------------------
            namespace omp2
            {
                namespace blocks
                {
                    //-----------------------------------------------------------------------------
                    //! The CPU OpenMP 2.0 block accelerator implementation details.
                    //-----------------------------------------------------------------------------
                    namespace detail
                    {
                        //#############################################################################
                        //! The CPU OpenMP 2.0 block accelerator.
                        //!
                        //! This accelerator allows parallel kernel execution on a CPU device.
                        //! It uses OpenMP 2.0 to implement the grid block parallelism.
                        //! The block size is restricted to 1x1x1.
                        //#############################################################################
                        template<
                            typename TDim,
                            typename TSize>
                        class AccCpuOmp2Blocks final :
                            public workdiv::WorkDivMembers<TDim, TSize>,
                            public idx::gb::IdxGbRef<TDim, TSize>,
                            public idx::bt::IdxBtZero<TDim, TSize>,
                            public atomic::AtomicNoOp,
                            public math::MathStl,
                            public block::shared::BlockSharedAllocNoSync
                        {
                        public:
                            friend class ::alpaka::exec::omp::omp2::blocks::detail::ExecCpuOmp2BlocksImpl<TDim, TSize>;

                        private:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp2Blocks(
                                TWorkDiv const & workDiv) :
                                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                                    idx::gb::IdxGbRef<TDim, TSize>(m_vuiGridBlockIdx),
                                    idx::bt::IdxBtZero<TDim, TSize>(),
                                    atomic::AtomicNoOp(),
                                    math::MathStl(),
                                    block::shared::BlockSharedAllocNoSync(),
                                    m_vuiGridBlockIdx(Vec<TDim, TSize>::zeros())
                            {}

                        public:
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp2Blocks(AccCpuOmp2Blocks const &) = delete;
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp2Blocks(AccCpuOmp2Blocks &&) = delete;
                            //-----------------------------------------------------------------------------
                            //! Copy assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuOmp2Blocks const &) -> AccCpuOmp2Blocks & = delete;
                            //-----------------------------------------------------------------------------
                            //! Move assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuOmp2Blocks &&) -> AccCpuOmp2Blocks & = delete;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~AccCpuOmp2Blocks() = default;

                            //-----------------------------------------------------------------------------
                            //! Syncs all threads in the current block.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads() const
                            -> void
                            {
                                // Nothing to do in here because only one thread in a group is allowed.
                            }

                            //-----------------------------------------------------------------------------
                            //! \return The pointer to the externally allocated block shared memory.
                            //-----------------------------------------------------------------------------
                            template<
                                typename T>
                            ALPAKA_FCT_ACC_NO_CUDA auto getBlockSharedExternMem() const
                            -> T *
                            {
                                return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                            }

                        private:
                            // getIdx
                            alignas(16u) Vec<TDim, TSize> mutable m_vuiGridBlockIdx;    //!< The index of the currently executed block.

                            // getBlockSharedExternMem
                            std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                        };
                    }
                }
            }
        }
    }

    template<
        typename TDim,
        typename TSize>
    using AccCpuOmp2Blocks = acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>;

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
            {
                using type = acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

                    return {
                        // m_uiMultiProcessorCount
                        static_cast<TSize>(1),
                        // m_uiBlockThreadsCountMax
                        static_cast<TSize>(1),
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim, TSize>::ones(),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max())};
                }
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp2Blocks<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 block accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 block accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
            {
                using type = exec::ExecCpuOmp2Blocks<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
