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
#include <alpaka/accs/omp/Idx.hpp>          // IdxOmp
#include <alpaka/accs/omp/Atomic.hpp>       // AtomicOmp

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
            //! The OpenMP4 accelerator.
            //-----------------------------------------------------------------------------
            namespace omp4
            {
                //-----------------------------------------------------------------------------
                //! The CPU OpenMP4 accelerator.
                //-----------------------------------------------------------------------------
                namespace cpu
                {
                    //-----------------------------------------------------------------------------
                    //! The CPU OpenMP4 accelerator implementation details.
                    //-----------------------------------------------------------------------------
                    namespace detail
                    {
                        template<
                            typename TDim>
                        class ExecCpuOmp4;
                        template<
                            typename TDim>
                        class ExecCpuOmp4Impl;

                        //#############################################################################
                        //! The CPU OpenMP4 accelerator.
                        //!
                        //! This accelerator allows parallel kernel execution on a CPU device.
                        //! It uses CPU OpenMP4 to implement the parallelism.
                        //#############################################################################
                        template<
                            typename TDim>
                        class AccCpuOmp4 final :
                            protected alpaka::workdiv::BasicWorkDiv<TDim>,
                            protected omp::detail::IdxOmp<TDim>,
                            protected omp::detail::AtomicOmp
                        {
                        public:
                            friend class ::alpaka::accs::omp::omp4::cpu::detail::ExecCpuOmp4Impl<TDim>;

                        private:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp4(
                                TWorkDiv const & workDiv) :
                                    alpaka::workdiv::BasicWorkDiv<TDim>(workDiv),
                                    omp::detail::IdxOmp<TDim>(m_vuiGridBlockIdx),
                                    AtomicOmp(),
                                    m_vuiGridBlockIdx(Vec<TDim>::zeros())
                            {}

                        public:
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp4(AccCpuOmp4 const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp4(AccCpuOmp4 &&) = delete;
#endif
                            //-----------------------------------------------------------------------------
                            //! Copy assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuOmp4 const &) -> AccCpuOmp4 & = delete;
                            //-----------------------------------------------------------------------------
                            //! Move assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuOmp4 &&) -> AccCpuOmp4 & = delete;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA ~AccCpuOmp4() noexcept = default;

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
                                    *static_cast<omp::detail::IdxOmp<TDim> const *>(this),
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
                                    addr,
                                    value,
                                    *static_cast<AtomicOmp const *>(this));
                            }

                            //-----------------------------------------------------------------------------
                            //! Syncs all threads in the current block.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads() const
                            -> void
                            {
                                #pragma omp barrier
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

                                // Assure that all threads have executed the return of the last allocBlockSharedMem function (if there was one before).
                                syncBlockThreads();

                                // Arbitrary decision: The thread with id 0 has to allocate the memory.
                                if(::omp_get_thread_num() == 0)
                                {
                                    // \TODO: C++14 std::make_unique would be better.
                                    m_vvuiSharedMem.emplace_back(
                                        std::unique_ptr<uint8_t[]>(
                                            reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                                }
                                syncBlockThreads();

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
    using AccCpuOmp4 = accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU OpenMP4 accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>;
            };
            //#############################################################################
            //! The CPU OpenMP4 accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    UInt const uiBlockThreadsCountMax(4u);
#else
                    // NOTE: ::omp_get_thread_limit() returns 2^31-1 (largest positive int value)...
                    //int const iThreadLimit(::omp_get_thread_limit());
                    //std::cout << BOOST_CURRENT_FUNCTION << " omp_get_thread_limit: " << iThreadLimit << std::endl;
                    // m_uiBlockThreadsCountMax
                    //UInt uiBlockThreadsCountMax(static_cast<UInt>(iThreadLimit));
                    UInt uiBlockThreadsCountMax(static_cast<UInt>(::omp_get_num_procs()));
#endif
                    return {
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim>::all(uiBlockThreadsCountMax),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim>::all(std::numeric_limits<typename Vec<TDim>::Val>::max())};
                }
            };
            //#############################################################################
            //! The CPU OpenMP4 accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccName<
                accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp4<" + std::to_string(TDim::value) + ">";
                }
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU OpenMP4 accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP4 accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU OpenMP4 accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = TDim;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU OpenMP4 accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = accs::omp::omp4::cpu::detail::ExecCpuOmp4<TDim>;
            };
        }
    }
}
