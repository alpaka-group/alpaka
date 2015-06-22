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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/dev/Traits.hpp>                // DevType
#include <alpaka/exec/Traits.hpp>               // ExecType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // DevCpu

#include <alpaka/core/OpenMp.hpp>

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused
#include <boost/align.hpp>                      // boost::aligned_alloc

#include <memory>                               // std::unique_ptr
#include <vector>                               // std::vector

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim>
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
                            typename TDim>
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
                            typename TDim>
                        class AccCpuOmp2Blocks final :
                            public workdiv::WorkDivMembers<TDim>,
                            public idx::gb::IdxGbRef<TDim>,
                            public idx::bt::IdxBtZero<TDim>,
                            public atomic::AtomicNoOp,
                            public math::MathStl
                        {
                        public:
                            friend class ::alpaka::exec::omp::omp2::blocks::detail::ExecCpuOmp2BlocksImpl<TDim>;

                        private:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp2Blocks(
                                TWorkDiv const & workDiv) :
                                    workdiv::WorkDivMembers<TDim>(workDiv),
                                    idx::gb::IdxGbRef<TDim>(m_vuiGridBlockIdx),
                                    idx::bt::IdxBtZero<TDim>(),
                                    atomic::AtomicNoOp(),
                                    math::MathStl(),
                                    m_vuiGridBlockIdx(Vec<TDim>::zeros())
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
                            //! \return Allocates block shared memory.
                            //-----------------------------------------------------------------------------
                            template<
                                typename T,
                                UInt TuiNumElements>
                            ALPAKA_FCT_ACC_NO_CUDA auto allocBlockSharedMem() const
                            -> T *
                            {
                                static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                                m_vvuiSharedMem.emplace_back(
                                    reinterpret_cast<uint8_t *>(
                                        boost::alignment::aligned_alloc(16u, sizeof(T) * TuiNumElements)));
                                return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
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
                            alignas(16u) Vec<TDim> mutable m_vuiGridBlockIdx;          //!< The index of the currently executed block.

                            // allocBlockSharedMem
                            std::vector<
                                std::unique_ptr<uint8_t, boost::alignment::aligned_delete>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                            // getBlockSharedExternMem
                            std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                        };
                    }
                }
            }
        }
    }

    template<
        typename TDim>
    using AccCpuOmp2Blocks = acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>;

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim>
                {
                    boost::ignore_unused(dev);

                    return {
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        1u,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim>::ones(),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim>::all(std::numeric_limits<typename Vec<TDim>::Val>::max())};
                }
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccName<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp2Blocks<" + std::to_string(TDim::value) + ">";
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
                typename TDim>
            struct DevType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
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
                typename TDim>
            struct DimType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
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
                typename TDim>
            struct ExecType<
                acc::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = exec::ExecCpuOmp2Blocks<TDim>;
            };
        }
    }
}
