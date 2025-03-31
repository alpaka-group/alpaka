
#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/platform/PlatformCpu.hpp"

#include <ranges>

namespace alpaka::trait
{

    namespace detail
    {
        template<typename T>
        concept ContiguousContainer = requires(T t) {
            t.size();
            t.data();
            requires std::ranges::contiguous_range<T>;
        };
    } // namespace detail

    //
    template<detail::ContiguousContainer TRange>
    struct DevType<TRange>
    {
        using type = DevCpu;
    };

    //
    template<detail::ContiguousContainer TRange>
    struct GetDev<TRange>
    {
        ALPAKA_FN_HOST static auto getDev(TRange const& /* view */) -> DevCpu
        {
            return getDevByIdx(PlatformCpu{}, 0u);
        }
    };

    //
    template<detail::ContiguousContainer TRange>
    struct DimType<TRange>
    {
        using type = DimInt<1u>;
    };

    //
    template<detail::ContiguousContainer TRange>
    struct ElemType<TRange>
    {
        using type = std::ranges::range_value_t<TRange>;
    };

    template<detail::ContiguousContainer TRange>
    struct GetExtents<TRange>
    {
        ALPAKA_FN_HOST constexpr auto operator()(TRange const& r) -> Vec<DimInt<1>, Idx<TRange>>
        {
            return {std::ranges::size(r)};
        }
    };

    template<detail::ContiguousContainer TRange>
    struct GetPtrNative<TRange>
    {
        ALPAKA_FN_HOST static auto getPtrNative(TRange const& view)
            -> std::add_pointer_t<std::add_const_t<std::ranges::range_value_t<TRange>>>
        {
            return std::data(view);
        }

        ALPAKA_FN_HOST static auto getPtrNative(TRange& view) -> std::add_pointer_t<std::ranges::range_value_t<TRange>>
        {
            return std::data(view);
        }
    };

    template<detail::ContiguousContainer TRange>
    struct GetOffsets<TRange>
    {
        ALPAKA_FN_HOST auto operator()(TRange const&) -> Vec<DimInt<1>, Idx<TRange>>
        {
            return {0};
        }
    };

    template<detail::ContiguousContainer TRange>
    struct IdxType<TRange>
    {
        using type = std::size_t;
    };
} // namespace alpaka::trait
