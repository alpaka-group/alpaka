/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/meta/Fold.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The extent traits.
    namespace trait
    {
        //! The extent get trait.
        //!
        //! If not specialized explicitly it returns 1.
        template<typename TIdxIntegralConst, typename TExtent, typename TSfinae = void>
        struct [[deprecated("Specialize GetExtents instead")]] GetExtent{
            ALPAKA_NO_HOST_ACC_WARNING ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const&)
                ->Idx<TExtent>{return static_cast<Idx<TExtent>>(1);
    } // namespace trait
}; // namespace alpaka

//! The GetExtents trait for getting the extents of an object as an alpaka::Vec.
template<typename TExtent, typename TSfinae = void>
struct GetExtents;

//! The extent set trait.
template<typename TIdxIntegralConst, typename TExtent, typename TExtentVal, typename TSfinae = void>
struct SetExtent;
} // namespace trait

//! \return The extent in the given dimension.
ALPAKA_NO_HOST_ACC_WARNING
template<std::size_t Tidx, typename TExtent>
[[deprecated("use getExtents(extent)[Tidx] instead")]] ALPAKA_FN_HOST_ACC auto getExtent(
    TExtent const& extent = TExtent()) -> Idx<TExtent>
{
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    return trait::GetExtent<DimInt<Tidx>, TExtent>::getExtent(extent);
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
}

//! \return The extents of the given object.
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto getExtents(T const& object) -> Vec<Dim<T>, Idx<T>>
{
    return trait::GetExtents<T>{}(object);
}

//! \tparam T has to specialize GetExtent.
//! \return The extents of the given object.
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
[[deprecated("use getExtents() instead")]] ALPAKA_FN_HOST_ACC auto constexpr getExtentVec(T const& object = {})
    -> Vec<Dim<T>, Idx<T>>
{
    return getExtents(object);
}

//! \tparam T has to specialize GetExtent.
//! \return The extent but only the last TDim elements.
ALPAKA_NO_HOST_ACC_WARNING
template<typename TDim, typename T>
ALPAKA_FN_HOST_ACC auto constexpr getExtentVecEnd(T const& object = {}) -> Vec<TDim, Idx<T>>
{
    static_assert(TDim::value <= Dim<T>::value, "Cannot get more items than the extent holds");

    auto const e = getExtents(object);
    Vec<TDim, Idx<T>> v{};
    if constexpr(TDim::value > 0)
        for(unsigned i = 0; i < TDim::value; i++)
            v[i] = e[(Dim<T>::value - TDim::value) + i];
    return v;
}

//! \return The width.
ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtent>
ALPAKA_FN_HOST_ACC auto getWidth(TExtent const& extent = TExtent()) -> Idx<TExtent>
{
    static_assert(Dim<TExtent>::value >= 1);
    return getExtents(extent)[Dim<TExtent>::value - 1u];
}
//! \return The height.
ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtent>
ALPAKA_FN_HOST_ACC auto getHeight(TExtent const& extent = TExtent()) -> Idx<TExtent>
{
    static_assert(Dim<TExtent>::value >= 2);
    return getExtents(extent)[Dim<TExtent>::value - 2u];
}
//! \return The depth.
ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtent>
ALPAKA_FN_HOST_ACC auto getDepth(TExtent const& extent = TExtent()) -> Idx<TExtent>
{
    static_assert(Dim<TExtent>::value >= 3);
    return getExtents(extent)[Dim<TExtent>::value - 3u];
}

//! \return The product of the extents of the given object.
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto getExtentProduct(T const& object) -> Idx<T>
{
    return getExtents(object).prod();
}

namespace trait
{
    //! The Vec extent get trait specialization.
    template<typename TDim, typename TVal>
    struct GetExtents<Vec<TDim, TVal>>
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC constexpr auto operator()(Vec<TDim, TVal> const& extent) const -> Vec<TDim, TVal>
        {
            return extent;
        }
    };
    //! The Vec extent set trait specialization.
    template<typename TIdxIntegralConst, typename TDim, typename TVal, typename TExtentVal>
    struct SetExtent<
        TIdxIntegralConst,
        Vec<TDim, TVal>,
        TExtentVal,
        std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static constexpr auto setExtent(Vec<TDim, TVal>& extent, TExtentVal const& extentVal)
            -> void
        {
            extent[TIdxIntegralConst::value] = extentVal;
        }
    };
} // namespace trait

//! Sets the extent in the given dimension.
ALPAKA_NO_HOST_ACC_WARNING
template<std::size_t Tidx, typename TExtent, typename TExtentVal>
ALPAKA_FN_HOST_ACC auto setExtent(TExtent& extent, TExtentVal const& extentVal) -> void
{
    trait::SetExtent<DimInt<Tidx>, TExtent, TExtentVal>::setExtent(extent, extentVal);
}
//! Sets the width.
ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtent, typename TWidth>
ALPAKA_FN_HOST_ACC auto setWidth(TExtent& extent, TWidth const& width) -> void
{
    setExtent<Dim<TExtent>::value - 1u>(extent, width);
}
//! Sets the height.
ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtent, typename THeight>
ALPAKA_FN_HOST_ACC auto setHeight(TExtent& extent, THeight const& height) -> void
{
    setExtent<Dim<TExtent>::value - 2u>(extent, height);
}
//! Sets the depth.
ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtent, typename TDepth>
ALPAKA_FN_HOST_ACC auto setDepth(TExtent& extent, TDepth const& depth) -> void
{
    setExtent<Dim<TExtent>::value - 3u>(extent, depth);
}

// Trait specializations for integral types.
namespace trait
{
    template<typename Integral>
    struct GetExtents<Integral, std::enable_if_t<std::is_integral_v<Integral>>>
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto operator()(Integral i) const
        {
            return Vec{i};
        }
    };
    //! The unsigned integral width set trait specialization.
    template<typename TExtent, typename TExtentVal>
    struct SetExtent<DimInt<0u>, TExtent, TExtentVal, std::enable_if_t<std::is_integral_v<TExtent>>>
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
        {
            extent = extentVal;
        }
    };
} // namespace trait
} // namespace alpaka
