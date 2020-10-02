/* Copyright 2019 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/core/Assert.hpp>
#include <alpaka/dev/DevOmp5.hpp>
#include <alpaka/queue/QueueOmp5Blocking.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <omp.h>

#include <memory>

namespace alpaka
{
    class DevOmp5;
    namespace buf
    {
        template<
            typename TElem,
            typename TDim,
            typename TIdx>
        class BufCpu;
    }
    namespace buf
    {
        namespace omp5
        {
            namespace detail
            {
                //#############################################################################
                //! The OMP5 memory buffer detail.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                class BufOmp5Impl
                {
                    static_assert(
                        !std::is_const<TElem>::value,
                        "The elem type of the buffer can not be const because the C++ Standard forbids containers of const elements!");
                    static_assert(
                        !std::is_const<TIdx>::value,
                        "The idx type of the buffer can not be const!");
                private:
                    using Elem = TElem;
                    using Dim = TDim;
                    //-----------------------------------------------------------------------------
                    //! Calculate the pitches purely from the extents.
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto calculatePitchesFromExtents(
                        TExtent const & extent)
                    -> Vec<TDim, TIdx>
                    {
                        Vec<TDim, TIdx> pitchBytes(Vec<TDim, TIdx>::all(0));
                        pitchBytes[TDim::value - 1u] = extent[TDim::value - 1u] * static_cast<TIdx>(sizeof(TElem));
                        for(TIdx i = TDim::value - 1u; i > static_cast<TIdx>(0u); --i)
                        {
                            pitchBytes[i-1] = extent[i-1] * pitchBytes[i];
                        }
                        return pitchBytes;
                    }

                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST BufOmp5Impl(
                        DevOmp5 const & dev,
                        TElem * const pMem,
                        TExtent const & extent) :
                            m_dev(dev),
                            m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                            m_pitchBytes(calculatePitchesFromExtents(m_extentElements)),
                            m_pMem(pMem)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            TDim::value == alpaka::Dim<TExtent>::value,
                            "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");
                        static_assert(
                            std::is_same<TIdx, Idx<TExtent>>::value,
                            "The idx type of TExtent and the TIdx template parameter have to be identical!");
                    }

                public:
                    DevOmp5 m_dev;
                    Vec<TDim, TIdx> m_extentElements;
                    Vec<TDim, TIdx> m_pitchBytes;
                    TElem* m_pMem;

                    BufOmp5Impl(const BufOmp5Impl&) = delete;
                    BufOmp5Impl(BufOmp5Impl &&) = default;
                    BufOmp5Impl& operator=(const BufOmp5Impl&) = delete;
                    BufOmp5Impl& operator=(BufOmp5Impl&&) = default;
                    ~BufOmp5Impl()
                    {
                        omp_target_free(m_pMem, m_dev.m_spDevOmp5Impl->iDevice());
                    }
                };
            }
        }
        template<
            typename TElem,
            typename TDim,
            typename TIdx>
        class BufOmp5
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor
            template<
                typename TExtent>
            ALPAKA_FN_HOST BufOmp5(
                DevOmp5 const & dev,
                TElem * const pMem,
                TExtent const & extent) :
                    m_spBufImpl(std::make_shared<omp5::detail::BufOmp5Impl<TElem, TDim, TIdx>>(dev, pMem, extent))
            {}

                BufOmp5(const BufOmp5&) = default;
                BufOmp5(BufOmp5 &&) = default;
            BufOmp5& operator=(const BufOmp5&) = default;
            BufOmp5& operator=(BufOmp5&&) = default;

            omp5::detail::BufOmp5Impl<TElem, TDim, TIdx>& operator*() {return *m_spBufImpl;}
            const omp5::detail::BufOmp5Impl<TElem, TDim, TIdx>& operator*() const {return *m_spBufImpl;}

            inline const Vec<TDim, TIdx>& extentElements() const {return m_spBufImpl->m_extentElements;}

        private:
            std::shared_ptr<omp5::detail::BufOmp5Impl<TElem, TDim, TIdx>> m_spBufImpl;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The BufOmp5 device type trait specialization.
        template<
            typename TElem,
            typename TDim,
            typename TIdx>
        struct DevType<
            buf::BufOmp5<TElem, TDim, TIdx>>
        {
            using type = DevOmp5;
        };
        //#############################################################################
        //! The BufOmp5 device get trait specialization.
        template<
            typename TElem,
            typename TDim,
            typename TIdx>
        struct GetDev<
            buf::BufOmp5<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(
                buf::BufOmp5<TElem, TDim, TIdx> const & buf)
            -> DevOmp5
            {
                return (*buf).m_dev;
            }
        };

        //#############################################################################
        //! The BufOmp5 dimension getter trait specialization.
        template<
            typename TElem,
            typename TDim,
            typename TIdx>
        struct DimType<
            buf::BufOmp5<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The BufOmp5 memory element type get trait specialization.
        template<
            typename TElem,
            typename TDim,
            typename TIdx>
        struct ElemType<
            buf::BufOmp5<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp5 extent get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                buf::BufOmp5<TElem, TDim, TIdx>,
                typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    buf::BufOmp5<TElem, TDim, TIdx> const & extent)
                -> TIdx
                {
                    return extent.extentElements()[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace view
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp5 native pointer get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetPtrNative<
                buf::BufOmp5<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getPtrNative(
                    buf::BufOmp5<TElem, TDim, TIdx> const & buf)
                -> TElem const *
                {
                    return (*buf).m_pMem;
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getPtrNative(
                    buf::BufOmp5<TElem, TDim, TIdx> & buf)
                -> TElem *
                {
                    return (*buf).m_pMem;
                }
            };
            //#############################################################################
            //! The BufOmp5 pointer on device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetPtrDev<
                buf::BufOmp5<TElem, TDim, TIdx>,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getPtrDev(
                    buf::BufOmp5<TElem, TDim, TIdx> const & buf,
                    DevOmp5 const & dev)
                -> TElem const *
                {
                    if(dev == getDev(buf))
                    {
                        return *buf.m_pMem;
                    }
                    else
                    {
                        throw std::runtime_error("The buffer is not accessible from the given device!");
                    }
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getPtrDev(
                    buf::BufOmp5<TElem, TDim, TIdx> & buf,
                    DevOmp5 const & dev)
                -> TElem *
                {
                    if(dev == getDev(buf))
                    {
                        return *buf.m_pMem;
                    }
                    else
                    {
                        throw std::runtime_error("The buffer is not accessible from the given device!");
                    }
                }
            };
            //#############################################################################
            //! The BufOmp5 pitch get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetPitchBytes<
                TIdxIntegralConst,
                buf::BufOmp5<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getPitchBytes(
                    buf::BufOmp5<TElem, TDim, TIdx> const & pitch)
                -> TIdx
                {
                    return (*pitch).m_pitchBytes[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace buf
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp5 1D memory allocation trait specialization.
            template<
                typename TElem,
                typename TIdx>
            struct Alloc<
                TElem,
                DimInt<1u>,
                TIdx,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                template<
                    typename TExtent>
                ALPAKA_FN_HOST static auto alloc(
                    DevOmp5 const & dev,
                    TExtent const & extent)
                -> buf::BufOmp5<TElem, DimInt<1u>, TIdx>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const width(extent::getWidth(extent));
                    auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

                    void * memPtr = omp_target_alloc(static_cast<std::size_t>(widthBytes), dev.m_spDevOmp5Impl->iDevice());

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                        << " ew: " << width
                        << " ewb: " << widthBytes
                        << " ptr: " << memPtr
                        << " device: " << dev.m_spDevOmp5Impl->iDevice()
                        << std::endl;
#endif
                    return
                        buf::BufOmp5<TElem, DimInt<1u>, TIdx>(
                            dev,
                            reinterpret_cast<TElem *>(memPtr),
                            extent);
                }
            };

            //#############################################################################
            //! The BufOmp5 nD memory allocation trait specialization. \todo Add pitch
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct Alloc<
                TElem,
                TDim,
                TIdx,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                template<
                    typename TExtent>
                ALPAKA_FN_HOST static auto alloc(
                    DevOmp5 const & dev,
                    TExtent const & extent)
                -> buf::BufOmp5<TElem, TDim, TIdx>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    std::size_t size = static_cast<std::size_t>(extent[0]*static_cast<TIdx>(sizeof(TElem)));
                    for (unsigned int a = 1u; a < static_cast<unsigned int>(TDim::value); ++a)
                        size *= static_cast<std::size_t>(extent[a]);

                    void * memPtr = omp_target_alloc(size, dev.m_spDevOmp5Impl->iDevice());
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                        << " dim: " << TDim::value
                        << " extent: " << static_cast<std::size_t>(extent::getExtentVec(extent).prod())*sizeof(TElem)
                        << " ewb: " << size
                        << " ptr: " << memPtr
                        << " device: " << dev.m_spDevOmp5Impl->iDevice()
                        << std::endl;
#endif
                    return
                        buf::BufOmp5<TElem, TDim, TIdx>(
                            dev,
                            reinterpret_cast<TElem *>(memPtr),
                            extent);
                }
            };

            //#############################################################################
            //! The BufOmp5 device memory mapping trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct Map<
                buf::BufOmp5<TElem, TDim, TIdx>,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto map(
                    buf::BufOmp5<TElem, TDim, TIdx> const & buf,
                    DevOmp5 const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(getDev(buf) != dev)
                    {
                        throw std::runtime_error("Mapping memory from one OMP5 device into an other OMP5 device not implemented!");
                    }
                    // If it is already the same device, nothing has to be mapped.
                }
            };
            //#############################################################################
            //! The BufOmp5 device memory unmapping trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct Unmap<
                buf::BufOmp5<TElem, TDim, TIdx>,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto unmap(
                    buf::BufOmp5<TElem, TDim, TIdx> const & buf,
                    DevOmp5 const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(getDev(buf) != dev)
                    {
                        throw std::runtime_error("Unmapping memory mapped from one OMP5 device into an other OMP5 device not implemented!");
                    }
                    // If it is already the same device, nothing has to be unmapped.
                }
            };
            //#############################################################################
            //! The BufOmp5 memory pinning trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct Pin<
                buf::BufOmp5<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto pin(
                    buf::BufOmp5<TElem, TDim, TIdx> &)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // No explicit pinning in OMP5? GPU would be pinned anyway.
                }
            };
            //#############################################################################
            //! The BufOmp5 memory unpinning trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct Unpin<
                buf::BufOmp5<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto unpin(
                    buf::BufOmp5<TElem, TDim, TIdx> &)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // No explicit pinning in OMP5? GPU would be pinned anyway.
                }
            };
            //#############################################################################
            //! The BufOmp5 memory pin state trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct IsPinned<
                buf::BufOmp5<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto isPinned(
                    buf::BufOmp5<TElem, TDim, TIdx> const &)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // No explicit pinning in OMP5? GPU would be pinned anyway.
                    return true;
                }
            };
            //#############################################################################
            //! The BufOmp5 memory prepareForAsyncCopy trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct PrepareForAsyncCopy<
                buf::BufOmp5<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto prepareForAsyncCopy(
                    buf::BufOmp5<TElem, TDim, TIdx> &)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // OMP5 device memory is always ready for async copy
                }
            };
        }
    }
    namespace traits
    {
        //#############################################################################
        //! The BufOmp5 offset get trait specialization.
        template<
            typename TIdxIntegralConst,
            typename TElem,
            typename TDim,
            typename TIdx>
        struct GetOffset<
            TIdxIntegralConst,
            buf::BufOmp5<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getOffset(
                buf::BufOmp5<TElem, TDim, TIdx> const &)
            -> TIdx
            {
                return 0u;
            }
        };

        //#############################################################################
        //! The BufOmp5 idx type trait specialization.
        template<
            typename TElem,
            typename TDim,
            typename TIdx>
        struct IdxType<
            buf::BufOmp5<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    namespace buf
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCpu CUDA device memory mapping trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct Map<
                buf::BufCpu<TElem, TDim, TIdx>,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto map(
                    buf::BufCpu<TElem, TDim, TIdx> & buf,
                    DevOmp5 const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(getDev(buf) != dev) //! \TODO WTF?
                    {
                        //   Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                        throw std::runtime_error("Mapping host memory to OMP5 device not implemented!");
                    }
                    // If it is already the same device, nothing has to be mapped.
                }
            };
            //#############################################################################
            //! The BufCpu CUDA device memory unmapping trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct Unmap<
                buf::BufCpu<TElem, TDim, TIdx>,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto unmap(
                    buf::BufCpu<TElem, TDim, TIdx> & buf,
                    DevOmp5 const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(getDev(buf) != dev) //! \TODO WTF?
                    {
                        throw std::runtime_error("Mapping host memory to OMP5 device not implemented!");
                    }
                    // If it is already the same device, nothing has to be unmapped.
                }
            };
        }
    }
    namespace view
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCpu pointer on CUDA device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetPtrDev<
                buf::BufCpu<TElem, TDim, TIdx>,
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getPtrDev(
                    buf::BufCpu<TElem, TDim, TIdx> const &,
                    DevOmp5 const &)
                -> TElem const *
                {
                    throw std::runtime_error("Mapping host memory to OMP5 device not implemented!");
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getPtrDev(
                    buf::BufCpu<TElem, TDim, TIdx> &,
                    DevOmp5 const &)
                -> TElem *
                {
                    throw std::runtime_error("Mapping host memory to OMP5 device not implemented!");
                }
            };
        }
    }
}

#include <alpaka/mem/buf/omp5/Copy.hpp>
#include <alpaka/mem/buf/omp5/Set.hpp>

#endif
