/* Copyright 2019 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/core/Assert.hpp>
#include <alpaka/dev/DevOmp4.hpp>
#include <alpaka/queue/QueueOmp4Blocking.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <omp.h>

#include <memory>

namespace alpaka
{
    namespace dev
    {
        class DevOmp4;
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufCpu;
        }
    }
    namespace mem
    {
        namespace buf
        {
            namespace omp4
            {
                namespace detail
                {
                    //#############################################################################
                    //! The OMP4 memory buffer detail.
                    template<
                        typename TElem,
                        typename TDim,
                        typename TIdx>
                    class BufOmp4Impl
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

                    public:
                        //-----------------------------------------------------------------------------
                        //! Constructor
                        template<
                            typename TExtent>
                        ALPAKA_FN_HOST BufOmp4Impl(
                            dev::DevOmp4 const & dev,
                            TElem * const pMem,
                            TExtent const & extent) :
                                m_dev(dev),
                                m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                                m_pMem(pMem)
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                            static_assert(
                                TDim::value == dim::Dim<TExtent>::value,
                                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");
                            static_assert(
                                std::is_same<TIdx, idx::Idx<TExtent>>::value,
                                "The idx type of TExtent and the TIdx template parameter have to be identical!");
                        }

                    public:
                        dev::DevOmp4 m_dev;
                        vec::Vec<TDim, TIdx> m_extentElements;
                        TElem* m_pMem;

                            BufOmp4Impl(const BufOmp4Impl&) = delete;
                            BufOmp4Impl(BufOmp4Impl &&) = default;
                        BufOmp4Impl& operator=(const BufOmp4Impl&) = delete;
                        BufOmp4Impl& operator=(BufOmp4Impl&&) = default;
                            ~BufOmp4Impl()
                        {
                            omp_target_free(m_pMem, m_dev.m_iDevice);
                        }
                    };
                }
            }
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufOmp4
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                template<
                    typename TExtent>
                ALPAKA_FN_HOST BufOmp4(
                    dev::DevOmp4 const & dev,
                    TElem * const pMem,
                    TExtent const & extent) :
                        m_spBufImpl(std::make_shared<omp4::detail::BufOmp4Impl<TElem, TDim, TIdx>>(dev, pMem, extent))
                {}

                    BufOmp4(const BufOmp4&) = default;
                    BufOmp4(BufOmp4 &&) = default;
                BufOmp4& operator=(const BufOmp4&) = default;
                BufOmp4& operator=(BufOmp4&&) = default;

                omp4::detail::BufOmp4Impl<TElem, TDim, TIdx>& operator*() {return *m_spBufImpl;}
                const omp4::detail::BufOmp4Impl<TElem, TDim, TIdx>& operator*() const {return *m_spBufImpl;}

                inline const vec::Vec<TDim, TIdx>& extentElements() const {return m_spBufImpl->m_extentElements;}

            private:
                std::shared_ptr<omp4::detail::BufOmp4Impl<TElem, TDim, TIdx>> m_spBufImpl;
            };
        }
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp4 device type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DevType<
                mem::buf::BufOmp4<TElem, TDim, TIdx>>
            {
                using type = dev::DevOmp4;
            };
            //#############################################################################
            //! The BufOmp4 device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetDev<
                mem::buf::BufOmp4<TElem, TDim, TIdx>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufOmp4<TElem, TDim, TIdx> const & buf)
                -> dev::DevOmp4
                {
                    return (*buf).m_dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp4 dimension getter trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DimType<
                mem::buf::BufOmp4<TElem, TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp4 memory element type get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct ElemType<
                mem::buf::BufOmp4<TElem, TDim, TIdx>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp4 extent get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                mem::buf::BufOmp4<TElem, TDim, TIdx>,
                typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufOmp4<TElem, TDim, TIdx> const & extent)
                -> TIdx
                {
                    return extent.extentElements()[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufOmp4 native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrNative<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> const & buf)
                    -> TElem const *
                    {
                        return (*buf).m_pMem;
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> & buf)
                    -> TElem *
                    {
                        return (*buf).m_pMem;
                    }
                };
                //#############################################################################
                //! The BufOmp4 pointer on device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> const & buf,
                        dev::DevOmp4 const & dev)
                    -> TElem const *
                    {
                        if(dev == dev::getDev(buf))
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
                        mem::buf::BufOmp4<TElem, TDim, TIdx> & buf,
                        dev::DevOmp4 const & dev)
                    -> TElem *
                    {
                        if(dev == dev::getDev(buf))
                        {
                            return *buf.m_pMem;
                        }
                        else
                        {
                            throw std::runtime_error("The buffer is not accessible from the given device!");
                        }
                    }
                };
            }
        }
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The BufOmp4 1D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<1u>,
                    TIdx,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevOmp4 const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufOmp4<TElem, dim::DimInt<1u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

                        void * memPtr = omp_target_alloc(widthBytes, dev.m_iDevice);
                        std::cerr << "alloc'd device ptr: " << memPtr << '\n';

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__
                            << " ew: " << width
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << std::endl;
#endif
                        return
                            mem::buf::BufOmp4<TElem, dim::DimInt<1u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                extent);
                    }
                };

                //#############################################################################
                //! The BufOmp4 nD memory allocation trait specialization. \todo Add pitch
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    TDim,
                    TIdx,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevOmp4 const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufOmp4<TElem, TDim, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto size = extent[0]*static_cast<TIdx>(sizeof(TElem));
                        for (TIdx a = 1u; a < TDim::value; ++a)
                            size *= extent[a];

                        void * memPtr = omp_target_alloc(size, dev.m_iDevice);
                        std::cerr << "alloc'd " << TDim::value << "D device ptr: " << memPtr << '\n';

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__
                            << " ew: " << volume/sizeof(TElem)
                            << " ewb: " << volume
                            << " ptr: " << memPtr
                            << std::endl;
#endif
                        return
                            mem::buf::BufOmp4<TElem, TDim, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                extent);
                    }
                };

                //#############################################################################
                //! The BufOmp4 device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> const & buf,
                        dev::DevOmp4 const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Mapping memory from one OMP4 device into an other OMP4 device not implemented!");
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufOmp4 device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> const & buf,
                        dev::DevOmp4 const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Unmapping memory mapped from one OMP4 device into an other OMP4 device not implemented!");
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
                //#############################################################################
                //! The BufOmp4 memory pinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Pin<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // No explicit pinning in OMP4? GPU would be pinned anyway.
                    }
                };
                //#############################################################################
                //! The BufOmp4 memory unpinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unpin<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // No explicit pinning in OMP4? GPU would be pinned anyway.
                    }
                };
                //#############################################################################
                //! The BufOmp4 memory pin state trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct IsPinned<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> const &)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // No explicit pinning in OMP4? GPU would be pinned anyway.
                        return true;
                    }
                };
                //#############################################################################
                //! The BufOmp4 memory prepareForAsyncCopy trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct PrepareForAsyncCopy<
                    mem::buf::BufOmp4<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto prepareForAsyncCopy(
                        mem::buf::BufOmp4<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // OMP4 device memory is always ready for async copy
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp4 offset get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetOffset<
                TIdxIntegralConst,
                mem::buf::BufOmp4<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                   mem::buf::BufOmp4<TElem, TDim, TIdx> const &)
                -> TIdx
                {
                    return 0u;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOmp4 idx type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct IdxType<
                mem::buf::BufOmp4<TElem, TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    namespace mem
    {
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
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevOmp4 const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev) //! \TODO WTF?
                        {
                            //   Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                            throw std::runtime_error("Mapping host memory to OMP4 device not implemented!");
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
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevOmp4 const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev) //! \TODO WTF?
                        {
                            throw std::runtime_error("Mapping host memory to OMP4 device not implemented!");
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
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> const &,
                        dev::DevOmp4 const &)
                    -> TElem const *
                    {
                        throw std::runtime_error("Mapping host memory to OMP4 device not implemented!");
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> &,
                        dev::DevOmp4 const &)
                    -> TElem *
                    {
                        throw std::runtime_error("Mapping host memory to OMP4 device not implemented!");
                    }
                };
            }
        }
    }
}

#include <alpaka/mem/buf/omp4/Copy.hpp>
#include <alpaka/mem/buf/omp4/Set.hpp>

#endif
