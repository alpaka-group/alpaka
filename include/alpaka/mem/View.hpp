/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/concepts/api.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/mem/BoundaryIter.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/mem/concepts/detail/InnerTypeAllowedCast.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/interface.hpp"

#include <cstdint>
#include <functional>

namespace alpaka
{
    /** @brief A non-owning view of a data region.
     *
     * This view is only holding a pointer to real data, copying the view is cheap.
     * Const-ness of the view instance is propagated to the data region.
     *
     * This satisfies the alpaka::concepts::IView concept and, therefore, also the alpaka::concepts::IMdSpan concept.
     */
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment = Alignment<>>
    struct View;

    /** @brief Create a view whose innermost dimension is contiguous.
     *
     * The API is obtained from @p anyWithApi. Pitches are calculated from the
     * extents and element type. The view does not own data.
     */
    template<typename T_ValueType, concepts::Alignment T_MemAlignment = Alignment<>>
    inline constexpr auto makeView(
        auto&& anyWithApi,
        T_ValueType* pointer,
        concepts::Vector auto const& extents,
        T_MemAlignment const memAlignment = T_MemAlignment{})
    {
        auto pitchMd = alpaka::calculatePitchesFromExtents<T_ValueType>(extents);
        return View{getApi(ALPAKA_FORWARD(anyWithApi)), pointer, extents, pitchMd, memAlignment};
    }

    /** @brief Create a view whose innermost dimension is contiguous.
     *
     * Use this overload for data whose layout cannot be described by pitches
     * calculated from the extents, such as a subregion of a larger allocation.
     * The API is obtained from @p anyWithApi; the view does not own the data.
     */
    template<typename T_ValueType, concepts::Alignment T_MemAlignment = Alignment<>>
    inline constexpr auto makeView(
        auto&& anyWithApi,
        T_ValueType* pointer,
        concepts::Vector auto const& extents,
        concepts::Vector auto const& pitches,
        T_MemAlignment const memAlignment = T_MemAlignment{})
    {
        static_assert(std::is_same_v<ALPAKA_TYPEOF(extents), ALPAKA_TYPEOF(pitches)>);
        return View{getApi(ALPAKA_FORWARD(anyWithApi)), pointer, extents, pitches, memAlignment};
    }

    /** @brief Create a view of @p any using an explicitly chosen API.
     *
     * Data, extents, pitches, and alignment are obtained from @p any. The view
     * refers to the same data and does not own the data. The @p api is derived from @p anyWithApi.
     */
    inline constexpr auto makeView(alpaka::concepts::Api auto&& anyWithApi, auto&& any)
    {
        return View{
            internal::getApi(ALPAKA_FORWARD(anyWithApi)),
            onHost::data(ALPAKA_FORWARD(any)),
            onHost::getExtents(ALPAKA_FORWARD(any)),
            onHost::getPitches(ALPAKA_FORWARD(any)),
            alpaka::getAlignment(ALPAKA_FORWARD(any))};
    }

    /** @brief Create a view whose innermost dimension is contiguous.
     *
     * The resulting view refers to the same data; it does not keep @p any alive.
     */
    inline constexpr auto makeView(auto&& any)
    {
        return View{
            internal::getApi(ALPAKA_FORWARD(any)),
            onHost::data(ALPAKA_FORWARD(any)),
            onHost::getExtents(ALPAKA_FORWARD(any)),
            onHost::getPitches(ALPAKA_FORWARD(any)),
            alpaka::getAlignment(ALPAKA_FORWARD(any))};
    }

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct View : MdSpan<T_Type, typename T_Extents::UniVec, typename T_Extents::UniVec, T_MemAlignment>
    {
    private:
        using BaseMdSpan = MdSpan<T_Type, typename T_Extents::UniVec, typename T_Extents::UniVec, T_MemAlignment>;

    public:
        /** Creates a view
         *
         * @param data handle to the physical data
         * @param extents n-dimensional extents in elements of the view. Must satisfy `n <= number_of_elements` in the
         * data handle.
         */
        template<
            alpaka::concepts::HasApi T_Any,
            alpaka::concepts::Vector T_UserExtents,
            alpaka::concepts::Vector T_UserPitches>
        constexpr View(
            T_Any const& any,
            T_Type* data,
            T_UserExtents const& extents,
            T_UserPitches const& pitches,
            T_MemAlignment const memAlignment = T_MemAlignment{})
            : BaseMdSpan{
                  data,
                  typename T_UserExtents::UniVec{extents},
                  typename T_UserPitches::UniVec{pitches},
                  memAlignment}
        {
            alpaka::unused(any);
            static_assert(
                isLosslesslyConvertible_v<typename T_UserPitches::value_type, typename T_UserExtents::value_type>,
                "extent type and pitch type must be lossless convertible");
        }

        template<typename T_TypeOther>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_TypeOther>
        constexpr View(View<T_Api, T_TypeOther, T_Extents, T_MemAlignment> const& other)
            : BaseMdSpan{static_cast<BaseMdSpan>(other)}
        {
        }

        constexpr View(View const&) = default;

        template<typename T_TypeOther>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_TypeOther>
        constexpr View(View<T_Api, T_TypeOther, T_Extents, T_MemAlignment>&& other)
            : BaseMdSpan{std::move(static_cast<BaseMdSpan>(other))}
        {
        }

        constexpr View(View&&) = default;

        /** Assignment operator keeping const-ness
         *
         * @attention the assign operator is not removing inner const-ness because the type signature is not changed.
         */
        constexpr View& operator=(View const&) = default;

        constexpr View& operator=(View&&) = default;

        static consteval T_Api getApi()
        {
            return T_Api{};
        }

        constexpr alpaka::concepts::IMdSpan auto getMdSpan() const
        {
            return BaseMdSpan::getConstMdSpan();
        }

        constexpr alpaka::concepts::IMdSpan auto getMdSpan()
        {
            return BaseMdSpan{*this};
        }

        /** create a read only view */
        constexpr auto getConstView() const
        {
            using ConstValueType = std::add_const_t<typename BaseMdSpan::value_type>;
            return View<T_Api, ConstValueType, T_Extents, T_MemAlignment>{
                T_Api{},
                static_cast<ConstValueType*>(this->data()),
                this->getExtents(),
                this->getPitches(),
                T_MemAlignment{}};
        }

        /** @brief Creates a sub view to a part of the memory.
         *
         * The sub view has the same dimension as the original.
         *
         * @param extents Number of elements for each dimension. Each number must be less than or equal to
         * the number of elements in the original dimension.
         * @return View which is pointing only to a part of the original view.
         *
         * @{
         */
        constexpr auto getSubView(alpaka::concepts::VectorOrScalar auto const& extents) const
        {
            static_assert(alpaka::trait::getDim_v<ALPAKA_TYPEOF(extents)> == T_Extents::dim());
            Vec extentMd = extents;
            assert((extentMd <= this->getExtents()).reduce(std::logical_and{}));
            return makeView(T_Api{}, this->data(), extentMd, this->getPitches(), T_MemAlignment{});
        }

        constexpr auto getSubView(alpaka::concepts::VectorOrScalar auto const& extents)
        {
            static_assert(alpaka::trait::getDim_v<ALPAKA_TYPEOF(extents)> == T_Extents::dim());
            Vec extentMd = extents;
            assert((extentMd <= this->getExtents()).reduce(std::logical_and{}));
            return makeView(T_Api{}, this->data(), extentMd, this->getPitches(), T_MemAlignment{});
        }

        /** @} */

        /** @brief Creates a sub view to a part of the memory.
         *
         * The sub view has the same dimension as the original. The offset defines the first coordinate of
         * each dimension. The `offset + extents - 1` defines the last element for each dimension in the
         * original view. Offset plus extents should not exceed the extents of the original view.
         *
         * @param offset offset in elements to the original view
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original view with a shifted origin pointer.
         *         The alignment of the sub view is reduced to the element alignment.
         *
         * @{
         */
        constexpr auto getSubView(
            alpaka::concepts::VectorOrScalar auto const& offset,
            alpaka::concepts::VectorOrScalar auto const& extents) const
        {
            static_assert(alpaka::trait::getDim_v<ALPAKA_TYPEOF(extents)> == T_Extents::dim());
            static_assert(alpaka::trait::getDim_v<ALPAKA_TYPEOF(offset)> == T_Extents::dim());

            Vec offsetMd = offset;
            Vec extentMd = extents;
            assert((offsetMd + extentMd <= this->getExtents()).reduce(std::logical_and{}));
            auto shiftedPtr = &(*this)[offsetMd];
            return makeView(T_Api{}, shiftedPtr, extentMd, this->getPitches(), Alignment<>{});
        }

        constexpr auto getSubView(
            alpaka::concepts::VectorOrScalar auto const& offset,
            alpaka::concepts::VectorOrScalar auto const& extents)
        {
            static_assert(alpaka::trait::getDim_v<ALPAKA_TYPEOF(extents)> == T_Extents::dim());
            static_assert(alpaka::trait::getDim_v<ALPAKA_TYPEOF(offset)> == T_Extents::dim());

            Vec offsetMd = offset;
            Vec extentMd = extents;
            assert((offsetMd + extentMd <= this->getExtents()).reduce(std::logical_and{}));
            auto shiftedPtr = &(*this)[offsetMd];
            return makeView(T_Api{}, shiftedPtr, extentMd, this->getPitches(), Alignment<>{});
        }

        /** @} */

        template<alpaka::concepts::Vector T_LowHaloVecType, alpaka::concepts::Vector T_UpHaloVecType>
        constexpr auto getSubView(
            alpaka::BoundaryDirection<View::dim(), T_LowHaloVecType, T_UpHaloVecType> boundaryDir) const
        {
            constexpr uint32_t dim = View::dim();
            auto offset = alpaka::Vec<uint32_t, dim>{};
            auto extents = alpaka::Vec<uint32_t, dim>{};

            for(uint32_t i = 0; i < dim; ++i)
            {
                switch(boundaryDir.data[i])
                {
                case BoundaryType::lower:
                    offset[i] = 0;
                    extents[i] = boundaryDir.lowerHaloSize[i];
                    break;
                case BoundaryType::upper:
                    offset[i] = this->getExtents()[i] - boundaryDir.upperHaloSize[i];
                    extents[i] = boundaryDir.upperHaloSize[i];
                    break;
                case BoundaryType::middle:
                    offset[i] = boundaryDir.lowerHaloSize[i];
                    extents[i] = this->getExtents()[i] - boundaryDir.lowerHaloSize[i] - boundaryDir.upperHaloSize[i];
                    break;
                default:
                    throw std::invalid_argument("invalid direction");
                }
            }
            return getSubView(offset, extents);
        }
    };

    template<typename T_Api, typename T_Type, concepts::Vector T_Extents, concepts::Alignment T_MemAlignment>
    std::ostream& operator<<(std::ostream& s, View<T_Api, T_Type, T_Extents, T_MemAlignment> const& view)
    {
        return s << "View{ dim=" << ALPAKA_TYPEOF(view)::dim() << ", api= " << onHost::getName(T_Api{})
                 << ", extents=" << view.getExtents().toString() << ", pitches=" << view.getPitches().toString()
                 << " , alignment=" << T_MemAlignment::template get<T_Type>() << " }";
    }

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches,
        alpaka::concepts::Alignment T_MemAlignment>
    ALPAKA_FN_DG View(
        T_Any const&,
        T_Type*,
        T_UserExtents const&,
        T_UserPitches const&,
        T_MemAlignment const memAlignment)
        -> View<ALPAKA_TYPEOF(getApi(std::declval<T_Any>())), T_Type, typename T_UserPitches::UniVec, T_MemAlignment>;

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches>
    ALPAKA_FN_DG View(T_Any, T_Type*, T_UserExtents const&, T_UserPitches const&)
        -> View<ALPAKA_TYPEOF(getApi(std::declval<T_Any>())), T_Type, typename T_UserPitches::UniVec, Alignment<>>;
} // namespace alpaka

namespace alpaka::internal
{
    // externally define the API trait to support constexpr evaluation
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<alpaka::View<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        inline constexpr auto operator()(auto&& view) const
        {
            alpaka::unused(view);
            return T_Api{};
        }
    };

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct CopyConstructableDataSource<View<T_Api, T_Type, T_Extents, T_MemAlignment>> : std::true_type
    {
        using InnerMutable = View<T_Api, std::remove_const_t<T_Type>, T_Extents, T_MemAlignment>;
        using InnerConst = View<T_Api, std::add_const_t<T_Type>, T_Extents, T_MemAlignment>;
    };
} // namespace alpaka::internal
