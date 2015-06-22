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
#include <alpaka/idx/bt/IdxBtOmp.hpp>           // IdxBtOmp
#include <alpaka/atomic/AtomicOmpCritSec.hpp>   // AtomicOmpCritSec
#include <alpaka/math/MathStl.hpp>              // MathStl

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/exec/Traits.hpp>               // ExecType
#include <alpaka/dev/Traits.hpp>                // DevType

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
        class ExecCpuOmp4;

        namespace omp
        {
            namespace omp4
            {
                namespace cpu
                {
                    namespace detail
                    {
                        template<
                            typename TDim>
                        class ExecCpuOmp4Impl;
                    }
                }
            }
        }
    }
    namespace acc
    {
        namespace omp
        {
            //-----------------------------------------------------------------------------
            //! The OpenMP4 accelerator.
            //-----------------------------------------------------------------------------
            namespace omp4
            {
                //-----------------------------------------------------------------------------
                //! The CPU OpenMP 4.0 accelerator.
                //-----------------------------------------------------------------------------
                namespace cpu
                {
                    //-----------------------------------------------------------------------------
                    //! The CPU OpenMP 4.0 accelerator implementation details.
                    //-----------------------------------------------------------------------------
                    namespace detail
                    {
                        //#############################################################################
                        //! The CPU OpenMP 4.0 accelerator.
                        //!
                        //! This accelerator allows parallel kernel execution on a CPU device.
                        //! It uses CPU OpenMP4 to implement the parallelism.
                        //#############################################################################
                        template<
                            typename TDim>
                        class AccCpuOmp4 final :
                            public workdiv::WorkDivMembers<TDim>,
                            public idx::gb::IdxGbRef<TDim>,
                            public idx::bt::IdxBtOmp<TDim>,
                            public atomic::AtomicOmpCritSec,
                            public math::MathStl
                        {
                        public:
                            friend class ::alpaka::exec::omp::omp4::cpu::detail::ExecCpuOmp4Impl<TDim>;

                        private:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp4(
                                TWorkDiv const & workDiv) :
                                    workdiv::WorkDivMembers<TDim>(workDiv),
                                    idx::gb::IdxGbRef<TDim>(m_vuiGridBlockIdx),
                                    idx::bt::IdxBtOmp<TDim>(),
                                    atomic::AtomicOmpCritSec(),
                                    math::MathStl(),
                                    m_vuiGridBlockIdx(Vec<TDim>::zeros())
                            {}

                        public:
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp4(AccCpuOmp4 const &) = delete;
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FCT_ACC_NO_CUDA AccCpuOmp4(AccCpuOmp4 &&) = delete;
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
                            ALPAKA_FCT_ACC_NO_CUDA ~AccCpuOmp4() = default;

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
                                    m_vvuiSharedMem.emplace_back(
                                        reinterpret_cast<uint8_t *>(
                                            boost::alignment::aligned_alloc(16u, sizeof(T) * TuiNumElements)));
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
                            alignas(16u) Vec<TDim> mutable m_vuiGridBlockIdx;            //!< The index of the currently executed block.

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
    using AccCpuOmp4 = acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>;

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>;
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
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
            //! The CPU OpenMP 4.0 accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccName<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp4<" + std::to_string(TDim::value) + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
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
            //! The CPU OpenMP 4.0 accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
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
            //! The CPU OpenMP 4.0 accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim>>
            {
                using type = exec::ExecCpuOmp4<TDim>;
            };
        }
    }
}
