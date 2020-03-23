/* Copyright 2019 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

#include <alpaka/core/Assert.hpp>
#include <alpaka/dev/DevOacc.hpp>
#include <alpaka/queue/QueueOaccBlocking.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <openacc.h>

#include <memory>

namespace alpaka
{
    namespace dev
    {
        class DevOacc;
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
            namespace oacc
            {
                namespace detail
                {
                    //#############################################################################
                    //! The OpenACC memory buffer detail.
                    template<
                        typename TElem,
                        typename TDim,
                        typename TIdx>
                    class BufOaccImpl
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
                        -> vec::Vec<TDim, TIdx>
                        {
                            vec::Vec<TDim, TIdx> pitchBytes(vec::Vec<TDim, TIdx>::all(0));
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
                        ALPAKA_FN_HOST BufOaccImpl(
                            dev::DevOacc const & dev,
                            TElem * const pMem,
                            TExtent const & extent) :
                                m_dev(dev),
                                m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                                m_pitchBytes(calculatePitchesFromExtents(m_extentElements)),
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
                        dev::DevOacc m_dev;
                        vec::Vec<TDim, TIdx> m_extentElements;
                        vec::Vec<TDim, TIdx> m_pitchBytes;
                        TElem* m_pMem;

                            BufOaccImpl(const BufOaccImpl&) = delete;
                            BufOaccImpl(BufOaccImpl &&) = default;
                        BufOaccImpl& operator=(const BufOaccImpl&) = delete;
                        BufOaccImpl& operator=(BufOaccImpl&&) = default;
                            ~BufOaccImpl()
                        {
                            m_dev.makeCurrent();
                            acc_free(m_pMem);
                        }
                    };
                }
            }
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufOacc
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                template<
                    typename TExtent>
                ALPAKA_FN_HOST BufOacc(
                    dev::DevOacc const & dev,
                    TElem * const pMem,
                    TExtent const & extent) :
                        m_spBufImpl(std::make_shared<oacc::detail::BufOaccImpl<TElem, TDim, TIdx>>(dev, pMem, extent))
                {}

                    BufOacc(const BufOacc&) = default;
                    BufOacc(BufOacc &&) = default;
                BufOacc& operator=(const BufOacc&) = default;
                BufOacc& operator=(BufOacc&&) = default;

                oacc::detail::BufOaccImpl<TElem, TDim, TIdx>& operator*() {return *m_spBufImpl;}
                const oacc::detail::BufOaccImpl<TElem, TDim, TIdx>& operator*() const {return *m_spBufImpl;}

                inline const vec::Vec<TDim, TIdx>& extentElements() const {return m_spBufImpl->m_extentElements;}

            private:
                std::shared_ptr<oacc::detail::BufOaccImpl<TElem, TDim, TIdx>> m_spBufImpl;
            };
        }
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufOacc device type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DevType<
                mem::buf::BufOacc<TElem, TDim, TIdx>>
            {
                using type = dev::DevOacc;
            };
            //#############################################################################
            //! The BufOacc device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetDev<
                mem::buf::BufOacc<TElem, TDim, TIdx>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufOacc<TElem, TDim, TIdx> const & buf)
                -> dev::DevOacc
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
            //! The BufOacc dimension getter trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DimType<
                mem::buf::BufOacc<TElem, TDim, TIdx>>
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
            //! The BufOacc memory element type get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct ElemType<
                mem::buf::BufOacc<TElem, TDim, TIdx>>
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
            //! The BufOacc extent get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                mem::buf::BufOacc<TElem, TDim, TIdx>,
                typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufOacc<TElem, TDim, TIdx> const & extent)
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
                //! The BufOacc native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrNative<
                    mem::buf::BufOacc<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufOacc<TElem, TDim, TIdx> const & buf)
                    -> TElem const *
                    {
                        return (*buf).m_pMem;
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufOacc<TElem, TDim, TIdx> & buf)
                    -> TElem *
                    {
                        return (*buf).m_pMem;
                    }
                };
                //#############################################################################
                //! The BufOacc pointer on device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufOacc<TElem, TDim, TIdx>,
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufOacc<TElem, TDim, TIdx> const & buf,
                        dev::DevOacc const & dev)
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
                        mem::buf::BufOacc<TElem, TDim, TIdx> & buf,
                        dev::DevOacc const & dev)
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
                //#############################################################################
                //! The BufOacc pitch get trait specialization.
                template<
                    typename TIdxIntegralConst,
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPitchBytes<
                    TIdxIntegralConst,
                    mem::buf::BufOacc<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::buf::BufOacc<TElem, TDim, TIdx> const & pitch)
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
                //! The BufOacc 1D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<1u>,
                    TIdx,
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevOacc const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufOacc<TElem, dim::DimInt<1u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

                        dev.makeCurrent();
                        void * memPtr = acc_malloc(static_cast<std::size_t>(widthBytes));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__
                            << " ew: " << width
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << " device: " << dev.m_spDevOaccImpl->iDevice()
                            << std::endl;
#endif
                        return
                            mem::buf::BufOacc<TElem, dim::DimInt<1u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                extent);
                    }
                };

                //#############################################################################
                //! The BufOacc nD memory allocation trait specialization. \todo Add pitch
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    TDim,
                    TIdx,
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevOacc const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufOacc<TElem, TDim, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        std::size_t size = static_cast<std::size_t>(extent[0]*static_cast<TIdx>(sizeof(TElem)));
                        for (unsigned int a = 1u; a < static_cast<unsigned int>(TDim::value); ++a)
                            size *= static_cast<std::size_t>(extent[a]);

                        dev.makeCurrent();
                        void * memPtr = acc_malloc(size);
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__ << "alloc'd " << TDim::value
                            << "D device ptr: " << memPtr << " on device " << dev.m_spDevOaccImpl->iDevice()
                            << " size " << size << " = " << static_cast<std::size_t>(extent::getExtentVec(extent).prod())*sizeof(TElem) << '\n';
#endif
                        return
                            mem::buf::BufOacc<TElem, TDim, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                extent);
                    }
                };

                //#############################################################################
                //! The BufOacc device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufOacc<TElem, TDim, TIdx>,
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufOacc<TElem, TDim, TIdx> const & buf,
                        dev::DevOacc const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Mapping memory from one OpenACC device into an other OpenACC device not implemented!");
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufOacc device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufOacc<TElem, TDim, TIdx>,
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufOacc<TElem, TDim, TIdx> const & buf,
                        dev::DevOacc const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Unmapping memory mapped from one OpenACC device into an other OpenACC device not implemented!");
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
                //#############################################################################
                //! The BufOacc memory pinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Pin<
                    mem::buf::BufOacc<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufOacc<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // No explicit pinning in OpenACC? GPU would be pinned anyway.
                    }
                };
                //#############################################################################
                //! The BufOacc memory unpinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unpin<
                    mem::buf::BufOacc<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufOacc<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // No explicit pinning in OpenACC? GPU would be pinned anyway.
                    }
                };
                //#############################################################################
                //! The BufOacc memory pin state trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct IsPinned<
                    mem::buf::BufOacc<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufOacc<TElem, TDim, TIdx> const &)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // No explicit pinning in OpenACC? GPU would be pinned anyway.
                        return true;
                    }
                };
                //#############################################################################
                //! The BufOacc memory prepareForAsyncCopy trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct PrepareForAsyncCopy<
                    mem::buf::BufOacc<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto prepareForAsyncCopy(
                        mem::buf::BufOacc<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // OpenACC device memory is always ready for async copy
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
            //! The BufOacc offset get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetOffset<
                TIdxIntegralConst,
                mem::buf::BufOacc<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                   mem::buf::BufOacc<TElem, TDim, TIdx> const &)
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
            //! The BufOacc idx type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct IdxType<
                mem::buf::BufOacc<TElem, TDim, TIdx>>
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
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevOacc const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev) //! \TODO WTF?
                        {
                            //   Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                            throw std::runtime_error("Mapping host memory to OpenACC device not implemented!");
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
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevOacc const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev) //! \TODO WTF?
                        {
                            throw std::runtime_error("Mapping host memory to OpenACC device not implemented!");
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
                    dev::DevOacc>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> const &,
                        dev::DevOacc const &)
                    -> TElem const *
                    {
                        throw std::runtime_error("Mapping host memory to OpenACC device not implemented!");
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> &,
                        dev::DevOacc const &)
                    -> TElem *
                    {
                        throw std::runtime_error("Mapping host memory to OpenACC device not implemented!");
                    }
                };
            }
        }
    }
}

#include <alpaka/mem/buf/oacc/Copy.hpp>
#include <alpaka/mem/buf/oacc/Set.hpp>

#endif
