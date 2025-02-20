/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The generic memory buffer template implementing muting accessors.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    class GenericBuf : public TBuf<TElem, TDim, TIdx>
    {
    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST GenericBuf(TDev const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : TBuf<TElem, TDim, TIdx>{
                  std::make_shared<TBufImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
        {
        }
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The GenericBuf device type trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DevType<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TDev;
    };

    //! The GenericBuf device get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetDev<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf) -> TDev
        {
            return getDev(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The GenericBuf dimension getter trait.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DimType<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The GenericBuf memory element type get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct ElemType<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The GenericBuf width get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct GetExtents<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf)
        {
            return getExtents(TBuf<TElem, TDim, TIdx>{buf});
        }
    };

    //! The GenericBuf native pointer get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetPtrNative<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return getPtrNative(TBuf<TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>& buf) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(getPtrNative(TBuf<TElem, TDim, TIdx>{buf}));
        }
    };

    //! The GenericBuf pointer on device get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct GetPtrDev<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>, TDev>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const& buf,
            TDev const& dev) -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return getPtrDev(TBuf<TElem, TDim, TIdx>{buf});
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }

        ALPAKA_FN_HOST static auto getPtrDev(GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>& buf, TDev const& dev)
            -> TElem*
        {
            if(dev == getDev(buf))
            {
                // cast away the TElem's constness from the ConstBuf's return type
                return const_cast<TElem*>(getPtrDev(TBuf<TElem, TDim, TIdx>{buf}));
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }
    };

    //! The GenericBuf offset get trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct GetOffsets<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx> const&) const
            -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The GenericBuf idx type trait specialization.
    template<
        template<typename, typename, typename> class TBuf,
        template<typename, typename, typename> class TBufImpl,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>

    struct IdxType<GenericBuf<TBuf, TBufImpl, TDev, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait
