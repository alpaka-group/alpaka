/* Copyright 2022 Alexander Matthes, Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>

namespace alpaka
{
    //! The CPU device handle.
    class DevCpu;

    //! The buffer traits.
    namespace trait
    {
        //! The memory buffer type trait.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TSfinae = void>
        struct BufType;

        //! The memory allocator trait.
        template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TSfinae = void>
        struct BufAlloc;

        //! The stream-ordered memory allocator trait.
        template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TSfinae = void>
        struct AsyncBufAlloc;

        //! The stream-ordered memory allocation capability trait.
        template<typename TDim, typename TDev>
        struct HasAsyncBufSupport : public std::false_type
        {
        };

        //! The pinned/mapped memory allocator trait.
        template<typename TPltf, typename TElem, typename TDim, typename TIdx>
        struct BufAllocMapped;

        //! The pinned/mapped memory allocation capability trait.
        template<typename TPltf>
        struct HasMappedBufSupport : public std::false_type
        {
        };
    } // namespace trait

    //! The memory buffer type trait alias template to remove the ::type.
    template<typename TDev, typename TElem, typename TDim, typename TIdx>
    using Buf = typename trait::BufType<alpaka::Dev<TDev>, TElem, TDim, TIdx>::type;

    //! Allocates memory on the given device.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TDev The type of device the buffer is allocated on.
    //! \param dev The device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx, typename TExtent, typename TDev>
#if(BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(12, 1, 0)) && defined(_OPENACC) && defined(NDEBUG)
    // Force O2 optimization level with GCC 12.1, OpenACC and Release mode.
    // See https://github.com/alpaka-group/alpaka/issues/1752
    [[gnu::optimize("O2")]]
#endif
    ALPAKA_FN_HOST auto
    allocBuf(TDev const& dev, TExtent const& extent = TExtent())
    {
        return trait::BufAlloc<TElem, Dim<TExtent>, TIdx, TDev>::allocBuf(dev, extent);
    }

    //! Allocates stream-ordered memory on the given device.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TQueue The type of queue used to order the buffer allocation.
    //! \param queue The queue used to order the buffer allocation.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx, typename TExtent, typename TQueue>
    ALPAKA_FN_HOST auto allocAsyncBuf(TQueue queue, TExtent const& extent = TExtent())
    {
        return trait::AsyncBufAlloc<TElem, Dim<TExtent>, TIdx, alpaka::Dev<TQueue>>::allocAsyncBuf(queue, extent);
    }

    /* TODO: Remove this pragma block once support for clang versions <= 13 is removed. These versions are unable to
       figure out that the template parameters are attached to a C++17 inline variable. */
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif
    //! Checks if the given device can allocate a stream-ordered memory buffer of the given dimensionality.
    //!
    //! \tparam TDev The type of device to allocate the buffer on.
    //! \tparam TDim The dimensionality of the buffer to allocate.
    template<typename TDev, typename TDim>
    constexpr inline bool hasAsyncBufSupport = trait::HasAsyncBufSupport<TDim, TDev>::value;
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    //! If supported, allocates stream-ordered memory on the given queue and the associated device.
    //! Otherwise, allocates regular memory on the device associated to the queue.
    //! Please note that stream-ordered and regular memory have different semantics:
    //! this function is provided for convenience in the cases where the difference is not relevant,
    //! and the stream-ordered memory is only used as a performance optimisation.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TQueue The type of queue used to order the buffer allocation.
    //! \param queue The queue used to order the buffer allocation.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx, typename TExtent, typename TQueue>
    ALPAKA_FN_HOST auto allocAsyncBufIfSupported(TQueue queue, TExtent const& extent = TExtent())
    {
        if constexpr(hasAsyncBufSupport<alpaka::Dev<TQueue>, Dim<TExtent>>)
        {
            return allocAsyncBuf<TElem, TIdx>(queue, extent);
        }
        else
        {
            return allocBuf<TElem, TIdx>(getDev(queue), extent);
        }

        ALPAKA_UNREACHABLE(allocBuf<TElem, TIdx>(getDev(queue), extent));
    }

    //! Allocates pinned/mapped host memory, accessible by all devices in the given platform.
    //!
    //! \tparam TPltf The platform from which the buffer is accessible.
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \param host The host device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TPltf, typename TElem, typename TIdx, typename TExtent>
    ALPAKA_FN_HOST auto allocMappedBuf(DevCpu const& host, TExtent const& extent = TExtent())
    {
        return trait::BufAllocMapped<TPltf, TElem, Dim<TExtent>, TIdx>::allocMappedBuf(host, extent);
    }

    /* TODO: Remove this pragma block once support for clang versions <= 13 is removed. These versions are unable to
       figure out that the template parameters are attached to a C++17 inline variable. */
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif
    //! Checks if the host can allocate a pinned/mapped host memory, accessible by all devices in the given platform.
    //!
    //! \tparam TPltf The platform from which the buffer is accessible.
    template<typename TPltf>
    constexpr inline bool hasMappedBufSupport = trait::HasMappedBufSupport<TPltf>::value;
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    //! If supported, allocates pinned/mapped host memory, accessible by all devices in the given platform.
    //! Otherwise, allocates regular host memory.
    //! Please note that pinned/mapped and regular memory may have different semantics:
    //! this function is provided for convenience in the cases where the difference is not relevant,
    //! and the pinned/mapped memory is only used as a performance optimisation.
    //!
    //! \tparam TPltf The platform from which the buffer is accessible.
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \param host The host device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TPltf, typename TElem, typename TIdx, typename TExtent>
    ALPAKA_FN_HOST auto allocMappedBufIfSupported(DevCpu const& host, TExtent const& extent = TExtent())
    {
        if constexpr(hasMappedBufSupport<TPltf>)
        {
            return allocMappedBuf<TPltf, TElem, TIdx>(host, extent);
        }
        else
        {
            return allocBuf<TElem, TIdx>(host, extent);
        }

        ALPAKA_UNREACHABLE(allocBuf<TElem, TIdx>(host, extent));
    }

    namespace detail
    {
        // TODO(bgruber): very crude
        template<typename DevDst, typename DevSrc>
        auto canZeroCopy(DevDst const& devDst, DevSrc const& devSrc) -> bool
        {
            if constexpr(std::is_same_v<DevDst, DevSrc>)
                if(devSrc == devDst)
                    return true;
            return false;
        }
    } // namespace detail

    //! Makes the content of the source view available on the device associated with the destination queue. If the
    //! destination shares the same memory space as the source view, no copy is performed and the destination view is
    //! updated to share the same buffer as the source view. Otherwise, a memcpy is performed from source to
    //! destination view.
    template<typename TQueue, typename TViewDst, typename TViewSrc>
    ALPAKA_FN_HOST void makeAvailable(TQueue& queue, TViewDst& viewDst, TViewSrc const& viewSrc)
    {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        if constexpr(std::is_same_v<TViewSrc, TViewDst>) // TODO(bgruber): lift this by converting buffer types
            if(detail::canZeroCopy(getDev(viewDst), getDev(viewSrc)))
            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::cout << "zero_memcopy: copy elided\n";
#endif
                viewDst = viewSrc;
                return;
            }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
        std::cout << "zero_memcopy: deep copy required\n";
#endif
        memcpy(queue, viewDst, viewSrc);
    }

    //! Makes the content of the source view available on the destination device. If the destination shares the same
    //! memory space as the source view, no copy is performed and the source view is returned. Otherwise a newly
    //! allocated buffer is created on the destination device and the content of the source view copied to it.
    template<
        typename TQueue,
        typename TDevDst,
        typename TViewSrc,
        std::enable_if_t<isDevice<TDevDst>, int> = 0,
        typename TViewDst = Buf<TDevDst, Elem<TViewSrc>, Dim<TViewSrc>, Idx<TViewSrc>>>
    ALPAKA_FN_HOST auto makeAvailable(TQueue& queue, TDevDst const& dstDev, TViewSrc const& viewSrc) -> TViewDst
    {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        if constexpr(std::is_same_v<TViewSrc, TViewDst>) // TODO(bgruber): lift this by converting buffer types
            if(detail::canZeroCopy(dstDev, getDev(viewSrc)))
            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::cout << "zero_memcopy: shallow copy returned\n";
#endif
                return viewSrc;
            }

        using E = Elem<TViewSrc>;
        using I = Idx<TViewSrc>;
        auto const extent = getExtentVec(viewSrc);
        TViewDst dst = [&]
        {
            using TDevQueue = Dev<TQueue>;
            if constexpr(std::is_same_v<TDevQueue, TDevDst>)
                if(getDev(queue) == dstDev)
                    return allocAsyncBufIfSupported<E, I>(queue, extent);
            return allocBuf<E, I>(dstDev, extent);
        }();
        memcpy(queue, dst, viewSrc);
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
        std::cout << "zero_memcopy: deep copy returned\n";
#endif
        return dst;
    }
} // namespace alpaka
