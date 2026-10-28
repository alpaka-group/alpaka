/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/api/host/Api.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/utility.hpp"

#include <ostream>

namespace alpaka
{

    /**
     * @brief An enum representing the different types of boundary, with LOWER, MIDDLE, and UPPER being valid states,
     * and OOB being invalid (out-of-bounds).
     */
    // public enum, the underlying type is part of the interface
    // NOLINTNEXTLINE(performance-enum-size)
    enum class BoundaryType : uint32_t
    {
        lower,
        middle,
        upper,
        oob
    };

    /**
     * @brief An n-dimensional boundary direction. Encodes a single unique boundary of an nD volume, e.g., a specific
     * corner of a 2D plane or a side of a 3D cube.
     *
     * See also: @ref BoundaryDirectionsContainer
     *
     * @tparam T_dim The dimensionality of the volume that this is a boundary direction for.
     * @tparam T_LowHaloVec The vector type used for the lower halo sizes.
     * @tparam T_UpHaloVec The vector type used for the upper halo sizes.
     */
    template<uint32_t T_dim, concepts::Vector T_LowHaloVec, concepts::Vector T_UpHaloVec>
    struct BoundaryDirection
    {
        using BoundaryVec = Vec<BoundaryType, T_dim>;

        BoundaryVec data;
        T_LowHaloVec lowerHaloSize;
        T_UpHaloVec upperHaloSize;

        constexpr BoundaryDirection(
            concepts::Vector auto const& boundaries,
            T_LowHaloVec const& lowerHaloSizes,
            T_UpHaloVec const& upperHaloSizes)
            : data(boundaries)
            , lowerHaloSize(lowerHaloSizes)
            , upperHaloSize(upperHaloSizes)
        {
        }

        /** @brief The dimensionality of the whole volume that this is a boundary direction for. Not to be confused
         * with boundaryDimensionality().
         */
        [[nodiscard]] static constexpr uint32_t dim()
        {
            return T_dim;
        }

        /** @brief The dimensionality of the boundary direction. For example, a vertex (corner) of a 3D-volume (cube)
         * is 0-dimensional. See also the functions isVertex(), isEdge(), etc.
         */
        [[nodiscard]] constexpr uint32_t boundaryDimensionality() const
        {
            uint32_t c = 0;
            for(uint32_t i = 0; i < T_dim; ++i)
            {
                if(data[i] == BoundaryType::middle)
                    ++c;
            }
            return c;
        }

        /** @brief Return true if this boundary direction describes a vertex, for example the corner of a plane.
         */
        [[nodiscard]] constexpr bool isVertex() const
        {
            return boundaryDimensionality() == 0;
        }

        /** @brief Return true if this boundary direction describes an edge, for example any of the 12 edges of a cube.
         */
        [[nodiscard]] constexpr bool isEdge() const
        {
            return boundaryDimensionality() == 1;
        }

        /** @brief Return true if this boundary direction describes a face, for example any of the 6 sides of a cube.
         */
        [[nodiscard]] constexpr bool isFace() const
        {
            return boundaryDimensionality() == 2;
        }

        /** @brief Return true if this boundary direction describes a cell, for example the interior of a cube or one
         * of the 8 cells in a tesseract.
         */
        [[nodiscard]] constexpr bool isCell() const
        {
            return boundaryDimensionality() == 3;
        }

        /** @brief Return true if this boundary direction describes the interior of a volume, like the 2D interior of a
         * plane or the 3D interior of a cube.
         */
        [[nodiscard]] constexpr bool isInterior() const
        {
            return boundaryDimensionality() == dim();
        }

        [[nodiscard]] constexpr auto operator<=>(BoundaryDirection const&) const = default;
    };

    /**
     * @brief The iterator type for @ref BoundaryDirectionsContainer.
     *
     * @tparam T_dim The dimensionality of the volume that this is a boundary direction iterator for.
     * @tparam T_LowHaloVec The vector type used for the lower halo sizes.
     * @tparam T_UpHaloVec The vector type used for the upper halo sizes.
     */
    template<uint32_t T_dim, concepts::Vector T_LowHaloVec, concepts::Vector T_UpHaloVec>
    struct BoundaryDirectionIter
    {
        using BoundaryVec = Vec<BoundaryType, T_dim>;

        using difference_type = std::ptrdiff_t;
        using value_type = BoundaryDirection<T_dim, T_LowHaloVec, T_UpHaloVec>;
        using reference = value_type&;
        using const_reference = value_type const&;
        using pointer = value_type*;
        using const_pointer = value_type const*;

        constexpr BoundaryDirectionIter(
            BoundaryVec const& boundaries,
            T_LowHaloVec const& lowerHaloSizes,
            T_UpHaloVec const& upperHaloSizes)
            : m_boundaries(boundaries, lowerHaloSizes, upperHaloSizes)
            , m_lowerHaloSizes(lowerHaloSizes)
            , m_upperHaloSizes(upperHaloSizes)
        {
        }

        [[nodiscard]] constexpr const_reference& operator*() const
        {
            return m_boundaries;
        }

        [[nodiscard]] constexpr reference& operator*()
        {
            return m_boundaries;
        }

        constexpr auto& operator++()
        {
            uint32_t i = T_dim - 1;
            bool oob = true;
            while(i != static_cast<uint32_t>(-1))
            {
                switch(m_boundaries.data[i])
                {
                case BoundaryType::lower:
                    m_boundaries.data[i] = BoundaryType::middle;
                    i = static_cast<uint32_t>(-1);
                    oob = false;
                    break;
                case BoundaryType::middle:
                    m_boundaries.data[i] = BoundaryType::upper;
                    i = static_cast<uint32_t>(-1);
                    oob = false;
                    break;
                case BoundaryType::upper:
                    m_boundaries.data[i] = BoundaryType::lower;
                    --i;
                    break;
                case BoundaryType::oob:
                    [[fallthrough]];
                default:
                    constexpr bool onHost = std::is_same_v<api::Host, ALPAKA_TYPEOF(thisApi())>;
                    if constexpr(onHost)
                        assert(false);
                    else
                        ALPAKA_ASSERT_ACC(false);
                }
            }
            if(oob)
            {
                m_boundaries
                    = {Vec<BoundaryType, T_dim>([](int) { return BoundaryType::oob; }),
                       m_lowerHaloSizes,
                       m_upperHaloSizes};
            }
            return *this;
        }

        [[nodiscard]] static consteval auto dim()
        {
            return T_dim;
        }

        [[nodiscard]] constexpr auto operator<=>(BoundaryDirectionIter const&) const = default;

    private:
        BoundaryDirection<T_dim, T_LowHaloVec, T_UpHaloVec> m_boundaries;

        T_LowHaloVec m_lowerHaloSizes;
        T_UpHaloVec m_upperHaloSizes;
    };

    /**
     * @brief A container for boundary directions of an n-dimensional volume.
     *
     * This class implements `begin()`, `end()`, and `length()`, and can be iterated over. This is useful for stencil
     * codes, where boundary conditions exist, that need to only be applied to elements on the borders of memory. To
     * create a BoundaryDirectionsContainer for a memory object, see @ref makeBoundaryDirIterator.
     *
     * A 0D boundary direction is a single value, a 1D boundary direction is a "line", for example edges of a cube, a
     * 2D boundary direction is a "plane", for example the sides of a cube.
     *
     * For example, a 1-dimensional (1D) volume has two 0D ends and a 1D center. A 2D volume has 4 0D corners, 4 1D
     * edges, and one 2D center. In general, there are 3^n boundaries for an nD volume.
     *
     * @tparam T_dim The dimensionality of the volume that this contains boundaries for.
     * @tparam T_LowHaloVec The vector type used for the lower halo sizes.
     * @tparam T_UpHaloVec The vector type used for the upper halo sizes.
     */
    template<uint32_t T_dim, concepts::Vector T_LowHaloVec, concepts::Vector T_UpHaloVec>
    struct BoundaryDirectionsContainer
    {
        static_assert(T_dim > 0, "0 Dimension Boundary Direction Container is not defined");

        constexpr BoundaryDirectionsContainer(T_LowHaloVec const& lowerHaloSizes, T_UpHaloVec const& upperHaloSizes)
            : m_lowerHaloSizes(lowerHaloSizes)
            , m_upperHaloSizes(upperHaloSizes)
        {
        }

        [[nodiscard]] constexpr BoundaryDirectionIter<T_dim, T_LowHaloVec, T_UpHaloVec> begin() const
        {
            return BoundaryDirectionIter<T_dim, T_LowHaloVec, T_UpHaloVec>{
                Vec<BoundaryType, T_dim>([](int) { return BoundaryType::lower; }),
                m_lowerHaloSizes,
                m_upperHaloSizes};
        }

        [[nodiscard]] constexpr BoundaryDirectionIter<T_dim, T_LowHaloVec, T_UpHaloVec> end() const
        {
            return BoundaryDirectionIter<T_dim, T_LowHaloVec, T_UpHaloVec>{
                Vec<BoundaryType, T_dim>([](int) { return BoundaryType::oob; }),
                m_lowerHaloSizes,
                m_upperHaloSizes};
        }

        [[nodiscard]] static consteval uint32_t length()
        {
            return ipow(3u, T_dim);
        }

        [[nodiscard]] static consteval auto dim()
        {
            return T_dim;
        }

    private:
        T_LowHaloVec const m_lowerHaloSizes;
        T_UpHaloVec const m_upperHaloSizes;
    };

    template<concepts::Vector T_LowHaloVecType, concepts::Vector T_UpHaloVecType>
    BoundaryDirectionsContainer(T_LowHaloVecType const& lowerHalos, T_UpHaloVecType const& upperHalos)
        -> BoundaryDirectionsContainer<T_LowHaloVecType::dim(), T_LowHaloVecType, T_UpHaloVecType>;

    /** @brief Construct and return a single @ref BoundaryDirection specifying the middle of a volume.
     */
    template<uint32_t T_dim>
    [[nodiscard]] constexpr auto makeCoreBoundaryDirection(
        concepts::Vector auto const& lowerHalos,
        concepts::Vector auto const& upperHalos)
    {
        return BoundaryDirection<T_dim, ALPAKA_TYPEOF(lowerHalos), ALPAKA_TYPEOF(upperHalos)>{
            fillCVec<BoundaryType, T_dim, BoundaryType::middle>(),
            lowerHalos,
            upperHalos};
    }

    /** @brief Construct and return a single @ref BoundaryDirection specifying the middle of a volume with symmetric
     * halos.
     */
    template<uint32_t T_dim>
    [[nodiscard]] constexpr auto makeCoreBoundaryDirection(concepts::Vector auto const& halos)
    {
        return BoundaryDirection<T_dim, ALPAKA_TYPEOF(halos), ALPAKA_TYPEOF(halos)>{
            fillCVec<BoundaryType, T_dim, BoundaryType::middle>(),
            halos,
            halos};
    }

    /**
     * @brief Construct and return a single @ref BoundaryDirection specifying the middle of a volume with all halo
     * sizes set to 1.
     */
    template<uint32_t T_dim>
    consteval auto makeCoreBoundaryDirection()
    {
        return makeCoreBoundaryDirection<T_dim>(fillCVec<uint32_t, T_dim, 1u>());
    }

    /** @brief Construct and return a @ref BoundaryDirectionsContainer. This container can be iterated over.
     *
     * This constructor uses a default halo size of 1 everywhere.
     *
     * @tparam T_dim The dimensionality of the container.
     */
    template<uint32_t T_dim>
    [[nodiscard]] constexpr auto makeBoundaryDirIterator()
    {
        auto lowerHalos = fillCVec<uint32_t, T_dim, static_cast<uint32_t>(1)>();
        auto upperHalos = fillCVec<uint32_t, T_dim, static_cast<uint32_t>(1)>();
        return BoundaryDirectionsContainer{lowerHalos, upperHalos};
    }

    /** @brief Construct and return a boundary direction container with the given halo sizes.
     * This container can be iterated over. See BoundaryDirectionsContainer.
     * The dimensionality is inferred from the given haloSizes.
     *
     * @param haloSizes The halo sizes per dimension. The halos are used for both "ends" of each dimension
     * symmetrically.
     */
    [[nodiscard]] constexpr auto makeBoundaryDirIterator(concepts::Vector auto const& haloSizes)
    {
        return BoundaryDirectionsContainer{haloSizes, haloSizes};
    }

    /** @brief Construct and return a @ref BoundaryDirectionsContainer with the given halo sizes.
     * This container can be iterated over.
     * The dimensionality is inferred from the given halo sizes, which are asserted to be identical.
     *
     * @param lowerHaloSizes The lower end halo sizes per dimension. These are the halos from 0 in each dimension.
     * @param upperHaloSizes The upper end halo sizes per dimension. These are the halos to `size()` in each dimension.
     */
    [[nodiscard]] constexpr auto makeBoundaryDirIterator(
        concepts::Vector auto const& lowerHaloSizes,
        concepts::Vector auto const& upperHaloSizes)
    {
        static_assert(
            ALPAKA_TYPEOF(lowerHaloSizes)::dim() == ALPAKA_TYPEOF(upperHaloSizes)::dim(),
            "dimension mismatch");
        return BoundaryDirectionsContainer{lowerHaloSizes, upperHaloSizes};
    }

    /** @brief Construct and return a @ref BoundaryDirectionsContainer for the given view with default (size 1) halo
     * sizes. This container can be iterated over.
     * For custom halo sizes, use one of the other overloads.
     *
     * @param view The given view; only the dimension of the view matters.
     */
    [[nodiscard]] constexpr auto makeBoundaryDirIterator(concepts::IView auto const& view)
    {
        return makeBoundaryDirIterator<static_cast<uint32_t>(ALPAKA_TYPEOF(view)::dim())>();
    }

    namespace trait
    {
        template<typename T>
        struct IsBoundaryDirection : std::false_type
        {
        };

        template<uint32_t T_dim, concepts::Vector T_LowHaloVec, concepts::Vector T_UpHaloVec>
        requires(T_dim == T_LowHaloVec::dim() && T_dim == T_UpHaloVec::dim())
        struct IsBoundaryDirection<BoundaryDirection<T_dim, T_LowHaloVec, T_UpHaloVec>> : std::true_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isBoundaryDirection_v = trait::IsBoundaryDirection<T>::value;

    namespace concepts
    {
        /** @brief Concept checking whether T is a boundary direction.
         */
        template<typename T>
        concept BoundaryDirection = isBoundaryDirection_v<T>;
    } // namespace concepts

    std::ostream& operator<<(std::ostream& os, concepts::BoundaryDirection auto const& bd)
    {
        for(uint32_t i = 0; i < bd.dim(); ++i)
        {
            switch(bd.data[i])
            {
            case BoundaryType::lower:
                os << 'v';
                break;
            case BoundaryType::middle:
                os << '-';
                break;
            case BoundaryType::upper:
                os << '^';
                break;
            case BoundaryType::oob:
                [[fallthrough]];
            default:
                os << 'x';
                break;
            }
        }

        if(bd.isVertex())
            os << " (vertex) ";
        if(bd.isEdge())
            os << " (edge)   ";
        if(bd.isFace())
            os << " (face)   ";
        if(bd.isCell())
            os << " (cell)   ";
        if(bd.boundaryDimensionality() >= 4)
            os << " (" << bd.boundaryDimensionality() << "D volume)";

        return os;
    }
} // namespace alpaka
