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
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
#include <alpaka/accs/serial/Idx.hpp>       // IdxSerial
#include <alpaka/accs/serial/Atomic.hpp>    // AtomicSerial

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Dev.hpp>            // DevType

// Implementation details.
#include <alpaka/accs/omp/Common.hpp>
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <memory>                           // std::unique_ptr
#include <vector>                           // std::vector

namespace alpaka
{
    namespace accs
    {
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
                        template<
                            typename TDim>
                        class ExecCpuOmp2Blocks;
                        template<
                            typename TDim>
                        class ExecCpuOmp2BlocksImpl;

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
                            protected workdiv::BasicWorkDiv<TDim>,
                            protected serial::detail::IdxSerial<TDim>,
                            protected serial::detail::AtomicSerial
                        {
                        public:
                            friend class ::alpaka::accs::omp::omp2::blocks::detail::ExecCpuOmp2BlocksImpl<TDim>;

                        private:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp2Blocks(
                                TWorkDiv const & workDiv) :
                                    workdiv::BasicWorkDiv<TDim>(workDiv),
                                    serial::detail::IdxSerial<TDim>(m_vuiGridBlockIdx),
                                    serial::detail::AtomicSerial(),
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
                            ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~AccCpuOmp2Blocks() noexcept = default;

                            //-----------------------------------------------------------------------------
                            //! \return The requested indices.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TOrigin,
                                typename TUnit>
                            ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                            -> Vec<TDim>
                            {
                                return idx::getIdx<TOrigin, TUnit>(
                                    *static_cast<serial::detail::IdxSerial<TDim> const *>(this),
                                    *static_cast<workdiv::BasicWorkDiv<TDim> const *>(this));
                            }

                            //-----------------------------------------------------------------------------
                            //! \return The requested extents.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TOrigin,
                                typename TUnit>
                            ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                            -> Vec<TDim>
                            {
                                return workdiv::getWorkDiv<TOrigin, TUnit>(
                                    *static_cast<workdiv::BasicWorkDiv<TDim> const *>(this));
                            }

                            //-----------------------------------------------------------------------------
                            //! Execute the atomic operation on the given address with the given value.
                            //! \return The old value before executing the atomic operation.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TOp,
                                typename T>
                            ALPAKA_FCT_ACC auto atomicOp(
                                T * const addr,
                                T const & value) const
                            -> T
                            {
                                return atomic::atomicOp<TOp, T>(
                                    *static_cast<AtomicSerial const *>(this),
                                    addr,
                                    value);
                            }

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

                                // \TODO: C++14 std::make_unique would be better.
                                m_vvuiSharedMem.emplace_back(
                                    std::unique_ptr<uint8_t[]>(
                                        reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
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
                            Vec<TDim> mutable m_vuiGridBlockIdx;                        //!< The index of the currently executed block.

                            // allocBlockSharedMem
                            std::vector<
                                std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                            // getBlockSharedExternMem
                            std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                        };
                    }
                }
            }
        }
    }

    template<
        typename TDim>
    using AccCpuOmp2Blocks = accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::DevCpu const & dev)
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
                accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp2Blocks<" + std::to_string(TDim::value) + ">";
                }
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = TDim;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::omp::omp2::blocks::detail::AccCpuOmp2Blocks<TDim>>
            {
                using type = accs::omp::omp2::blocks::detail::ExecCpuOmp2Blocks<TDim>;
            };
        }
    }
}
