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
#include <alpaka/block/shared/BlockSharedAllocMasterSync.hpp>  // BlockSharedAllocMasterSync

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/exec/Traits.hpp>               // ExecType
#include <alpaka/dev/Traits.hpp>                // DevType
#include <alpaka/size/Traits.hpp>               // size::SizeType

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
                            typename TDim,
                            typename TSize>
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
                            typename TDim,
                            typename TSize>
                        class AccCpuOmp4 final :
                            public workdiv::WorkDivMembers<TDim, TSize>,
                            public idx::gb::IdxGbRef<TDim, TSize>,
                            public idx::bt::IdxBtOmp<TDim, TSize>,
                            public atomic::AtomicOmpCritSec,
                            public math::MathStl,
                            public block::shared::BlockSharedAllocMasterSync
                        {
                        public:
                            friend class ::alpaka::exec::omp::omp4::cpu::detail::ExecCpuOmp4Impl<TDim, TSize>;

                        private:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp4(
                                TWorkDiv const & workDiv) :
                                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                                    idx::gb::IdxGbRef<TDim, TSize>(m_vuiGridBlockIdx),
                                    idx::bt::IdxBtOmp<TDim, TSize>(),
                                    atomic::AtomicOmpCritSec(),
                                    math::MathStl(),
                                    block::shared::BlockSharedAllocMasterSync(
                                        [this](){syncBlockThreads();},
                                        [](){return (::omp_get_thread_num() == 0);}),
                                    m_vuiGridBlockIdx(Vec<TDim, TSize>::zeros())
                            {}

                        public:
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp4(AccCpuOmp4 const &) = delete;
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp4(AccCpuOmp4 &&) = delete;
                            //-----------------------------------------------------------------------------
                            //! Copy assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp4 const &) -> AccCpuOmp4 & = delete;
                            //-----------------------------------------------------------------------------
                            //! Move assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp4 &&) -> AccCpuOmp4 & = delete;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA ~AccCpuOmp4() = default;

                            //-----------------------------------------------------------------------------
                            //! Syncs all threads in the current block.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA auto syncBlockThreads() const
                            -> void
                            {
                                #pragma omp barrier
                            }

                            //-----------------------------------------------------------------------------
                            //! \return The pointer to the externally allocated block shared memory.
                            //-----------------------------------------------------------------------------
                            template<
                                typename T>
                            ALPAKA_FN_ACC_NO_CUDA auto getBlockSharedExternMem() const
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
    using AccCpuOmp4 = acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>;

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
            {
                using type = acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    auto const uiBlockThreadsCountMax(static_cast<TSize>(4));
#else
                    // NOTE: ::omp_get_thread_limit() returns 2^31-1 (largest positive int value)...
                    auto const uiBlockThreadsCountMax(static_cast<TSize>(::omp_get_num_procs()));
#endif
                    return {
                        // m_uiMultiProcessorCount
                        static_cast<TSize>(1),
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim, TSize>::all(uiBlockThreadsCountMax),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max())};
                }
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
            {
                ALPAKA_FN_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp4<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
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
                typename TDim,
                typename TSize>
            struct DevType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
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
                typename TDim,
                typename TSize>
            struct DimType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
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
                typename TDim,
                typename TSize>
            struct ExecType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
            {
                using type = exec::ExecCpuOmp4<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::omp::omp4::cpu::detail::AccCpuOmp4<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
