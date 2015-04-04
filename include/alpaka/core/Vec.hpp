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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_ACC, ALPAKA_ALIGN
#include <alpaka/core/BasicDims.hpp>// dim::Dim<N>
#include <alpaka/core/IntegerSequence.hpp>    //

#include <alpaka/traits/Dim.hpp>    // traits::getDim
#include <alpaka/traits/Extent.hpp> // traits::getWidth, ...
#include <alpaka/traits/Offset.hpp> // traits::getOffsetX, ...

#include <boost/mpl/and.hpp>        // boost::mpl::and_
//#include <boost/type_traits/is_convertible.hpp>
#include <boost/predef.h>           // workarounds

#include <cstdint>                  // std::uint32_t
#include <ostream>                  // std::ostream
#include <cassert>                  // assert
#include <type_traits>              // std::enable_if

namespace alpaka
{
    //#############################################################################
    //! A n-dimensional vector.
    //#############################################################################
    // \TODO: Replace the integer template parameter by a dimension type.
    template<
        UInt TuiDim,
        typename TVal = UInt>
    class Vec
    {
    public:
        static_assert(TuiDim>0, "Size of the vector is required to be greater then zero!");

        static const UInt s_uiDim = TuiDim;
        using Val = TVal;

    public:
        //-----------------------------------------------------------------------------
        //! Default-constructor.
        //! Every value is set to zero.
        //-----------------------------------------------------------------------------
        /*ALPAKA_FCT_HOST_ACC Vec()
        {
            for(UInt i(0); i<TuiDim; ++i)
            {
                m_auiData[i] = 0;
            }
        }*/
        //-----------------------------------------------------------------------------
        //! Constructor.
        //! \param val The value every entry is set to.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC Vec(
            TVal const & val)
        {
            for(UInt i(0); i<TuiDim; ++i)
            {
                m_auiData[i] = val;
            }
        }
        //-----------------------------------------------------------------------------
        //! Value-constructor.
        //! This constructor is only available if the number of parameters matches the vector size.
        //-----------------------------------------------------------------------------
        template<
            typename TFirstArg,
            typename... TArgs,
            typename = typename std::enable_if<
                (sizeof...(TArgs) == (TuiDim-1))
                && std::is_convertible<typename std::decay<TFirstArg>::type, TVal>::value
                //&& boost::mpl::and_<boost::mpl::true_, boost::mpl::true_, std::is_convertible<typename std::decay<TArgs>::type, TVal>...>::value
            >::type>
        ALPAKA_FCT_HOST_ACC Vec(
            TFirstArg && val,
            TArgs && ... values)
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 22609))   // MSVC does not compile the basic array initialization: "error C2536: 'alpaka::Vec<0x03>::alpaka::Vec<0x03>::m_auiData': cannot specify explicit initializer for arrays"
            :
                m_auiData{std::forward<TFirstArg>(val), std::forward<TArgs>(values)...}
#endif
        {
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 22609))
            TVal auiData2[TuiDim] = {std::forward<TFirstArg>(val), std::forward<TArgs>(values)...};
            for(UInt i(0); i<TuiDim; ++i)
            {
                m_auiData[i] = auiData2[i];
            }
#endif
        }
        //-----------------------------------------------------------------------------
        //! Extents-constructor.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            UInt TuiDimSfinae = TuiDim,
            typename = typename std::enable_if<TuiDimSfinae == 1>::type>
        ALPAKA_FCT_HOST_ACC static auto fromExtents(
            TExtents const & extents)
        -> Vec<1u, TVal>
        {
            return Vec<1u, TVal>(
                extent::getWidth(extents));
        }
        //-----------------------------------------------------------------------------
        //! Extents-constructor.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            UInt TuiDimSfinae = TuiDim,
            typename = typename std::enable_if<TuiDimSfinae == 2>::type>
        ALPAKA_FCT_HOST_ACC static auto fromExtents(
            TExtents const & extents)
        -> Vec<2u, TVal>
        {
            return Vec<2u, TVal>(
                extent::getWidth(extents),
                extent::getHeight(extents));
        }
        //-----------------------------------------------------------------------------
        //! Extents-constructor.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            UInt TuiDimSfinae = TuiDim,
            typename = typename std::enable_if<TuiDimSfinae == 3>::type>
        ALPAKA_FCT_HOST_ACC static auto fromExtents(
            TExtents const & extents)
        -> Vec<3u, TVal>
        {
            return Vec<3u, TVal>(
                extent::getWidth(extents),
                extent::getHeight(extents),
                extent::getDepth(extents));
        }
        //-----------------------------------------------------------------------------
        //! Offsets-constructor.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            UInt TuiDimSfinae = TuiDim,
            typename = typename std::enable_if<TuiDimSfinae == 1>::type>
        ALPAKA_FCT_HOST_ACC static auto fromOffsets(
            TOffsets const & offsets)
        -> Vec<1u, TVal>
        {
            return Vec<1u, TVal>(
                offset::getOffsetX(offsets));
        }
        //-----------------------------------------------------------------------------
        //! Offsets-constructor.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            UInt TuiDimSfinae = TuiDim,
            typename = typename std::enable_if<TuiDimSfinae == 2>::type>
        ALPAKA_FCT_HOST_ACC static auto fromOffsets(
            TOffsets const & offsets)
        -> Vec<2u, TVal>
        {
            return Vec<2u, TVal>(
                offset::getOffsetX(offsets),
                offset::getOffsetY(offsets));
        }
        //-----------------------------------------------------------------------------
        //! Offsets-constructor.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            UInt TuiDim2 = TuiDim,
            typename = typename std::enable_if<TuiDim2 == 3>::type>
        ALPAKA_FCT_HOST_ACC static auto fromOffsets(
            TOffsets const & offsets)
        -> Vec<3u, TVal>
        {
            return Vec<3u, TVal>(
                offset::getOffsetX(offsets),
                offset::getOffsetY(offsets),
                offset::getOffsetZ(offsets));
        }
        //-----------------------------------------------------------------------------
        //! Copy constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC Vec(Vec const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
        //-----------------------------------------------------------------------------
        //! Move constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC Vec(Vec &&) = default;
#endif
        //-----------------------------------------------------------------------------
        //! Copy assignment.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator=(Vec const &) -> Vec & = default;
        //-----------------------------------------------------------------------------
        //! Destructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC virtual ~Vec() noexcept = default;

        //-----------------------------------------------------------------------------
        //! Destructor.
        //-----------------------------------------------------------------------------
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        ALPAKA_FCT_HOST_ACC auto static getDim()
#else
        ALPAKA_FCT_HOST_ACC auto static constexpr getDim()
#endif
        -> UInt
        {
            return TuiDim;
        }

        //-----------------------------------------------------------------------------
        //! \return The sub-vector consisting of the first N elements of the source vector.
        //-----------------------------------------------------------------------------
        template<
            UInt TuiSubDim>
        ALPAKA_FCT_HOST_ACC auto subVec() const
        -> Vec<TuiSubDim, TVal>
        {
            static_assert(TuiSubDim <= TuiDim, "The sub-vector has to be smaller (or same size) then the origin vector.");

#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            using IdxSequence = typename alpaka::detail::make_index_sequence<TuiSubDim>::type;
#else
            using IdxSequence = alpaka::detail::make_index_sequence<TuiSubDim>;
#endif
            return subVecFromIndices(IdxSequence());
        }
        //-----------------------------------------------------------------------------
        //! \return The sub-vector consisting of the elements specified by the indices.
        //-----------------------------------------------------------------------------
        template<
            size_t... TIndices>
        ALPAKA_FCT_HOST_ACC auto subVecFromIndices(
#if !BOOST_COMP_MSVC     // MSVC 190022512 introduced a new bug with alias templates: error C3520: 'TIndices': parameter pack must be expanded in this context
            detail::index_sequence<TIndices...> const &) const
#else
            detail::integer_sequence<std::size_t, TIndices...> const &) const
#endif
        -> Vec<sizeof...(TIndices), TVal>
        {
            static_assert(sizeof...(TIndices) <= TuiDim, "The sub-vector has to be smaller (or same size) then the origin vector.");

            return Vec<sizeof...(TIndices), TVal>((*this)[TIndices]...);
        }

        //-----------------------------------------------------------------------------
        //! \return A reference to the value at the given index.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator[](
            UInt const uiIdx)
        -> TVal &
        {
            assert(uiIdx<TuiDim);
            return m_auiData[uiIdx];
        }
        //-----------------------------------------------------------------------------
        //! \return The value at the given index.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator[](
            UInt const uiIdx) const
        -> TVal
        {
            assert(uiIdx<TuiDim);
            return m_auiData[uiIdx];
        }

        //-----------------------------------------------------------------------------
        // Equality comparison operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator==(
            Vec const & rhs) const
        -> bool
        {
            for(UInt i(0); i < TuiDim; i++)
            {
                if((*this)[i] != rhs[i])
                {
                    return TuiDim;
                }
            }
            return true;
        }
        //-----------------------------------------------------------------------------
        // Inequality comparison operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator!=(
            Vec const & rhs) const
        -> bool
        {
            return !((*this) == rhs);
        }

        //-----------------------------------------------------------------------------
        //! \return The product of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto prod() const
        -> TVal
        {
            TVal uiProd(m_auiData[0]);
            for(UInt i(1); i<TuiDim; ++i)
            {
                uiProd *= m_auiData[i];
            }
            return uiProd;
        }

        //-----------------------------------------------------------------------------
        //! Calculates the dot product of two vectors.
        //-----------------------------------------------------------------------------
        /*ALPAKA_FCT_HOST_ACC static auto dotProduct(
            Vec const & p,
            Vec const & q)
        -> TVal
        {
            TVal uiProd(0);
            for(size_t i(0); i<TuiDim; ++i)
            {
                uiProd += p[i] * q[i];
            }
            return uiProd;
        }*/

    private:
        ALPAKA_ALIGN(TVal, m_auiData[TuiDim]);
    };

    //-----------------------------------------------------------------------------
    //! \return The element wise sum of two vectors.
    //-----------------------------------------------------------------------------
    template<
        UInt TuiDim,
        typename TVal>
    ALPAKA_FCT_HOST_ACC auto operator+(
        Vec<TuiDim, TVal> const & p,
        Vec<TuiDim, TVal> const & q)
    -> Vec<TuiDim, TVal>
    {
        Vec<TuiDim, TVal> vRet(0u);
        for(UInt i(0); i<TuiDim; ++i)
        {
            vRet[i] = p[i] + q[i];
        }
        return vRet;
    }

    //-----------------------------------------------------------------------------
    //! \return The element wise product of two vectors.
    //-----------------------------------------------------------------------------
    template<
        UInt TuiDim,
        typename TVal>
    ALPAKA_FCT_HOST_ACC auto operator*(
        Vec<TuiDim, TVal> const & p,
        Vec<TuiDim, TVal> const & q)
    -> Vec<TuiDim, TVal>
    {
        Vec<TuiDim, TVal> vRet(0u);
        for(UInt i(0); i<TuiDim; ++i)
        {
            vRet[i] = p[i] * q[i];
        }
        return vRet;
    }

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<
        UInt TuiDim,
        typename TVal>
    ALPAKA_FCT_HOST auto operator<<(
        std::ostream & os,
        Vec<TuiDim, TVal> const & v)
    -> std::ostream &
    {
        os << "(";
        for(UInt i(0); i<TuiDim; ++i)
        {
            os << v[i];
            if(i<TuiDim-1)
            {
                os << ", ";
            }
        }
        os << ")";

        return os;
    }

    namespace detail
    {
        //#############################################################################
        //! The dimension to vector type transformation trait.
        //#############################################################################
        template<
            typename TDim>
        struct DimToVec
        {
            using type = Vec<TDim::value>;
        };
    }

    //#############################################################################
    //! The dimension to vector type alias template to remove the ::type.
    //#############################################################################
    template<
        typename TDim>
    using DimToVecT = typename detail::DimToVec<TDim>::type;

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The Vec<TuiDim> dimension get trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct DimType<
                alpaka::Vec<TuiDim, TVal>>
            {
                using type = alpaka::dim::Dim<TuiDim>;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The Vec<TuiDim> width get trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct GetWidth<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 1u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getWidth(
                    alpaka::Vec<TuiDim, TVal> const & extent)
                -> TVal
                {
                    return extent[0u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> width set trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct SetWidth<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 1u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto setWidth(
                    alpaka::Vec<TuiDim, TVal> & extent,
                    TVal const & width)
                -> void
                {
                    extent[0u] = width;
                }
            };

            //#############################################################################
            //! The Vec<TuiDim> height get trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct GetHeight<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 2u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getHeight(
                    alpaka::Vec<TuiDim, TVal> const & extent)
                -> TVal
                {
                    return extent[1u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> height set trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct SetHeight<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 2u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto setHeight(
                    alpaka::Vec<TuiDim, TVal> & extent,
                    TVal const & height)
                -> void
                {
                    extent[1u] = height;
                }
            };

            //#############################################################################
            //! The Vec<TuiDim> depth get trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct GetDepth<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 3u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getDepth(
                    alpaka::Vec<TuiDim, TVal> const & extent)
                -> TVal
                {
                    return extent[2u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> depth set trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct SetDepth<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 3u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto setDepth(
                    alpaka::Vec<TuiDim, TVal> & extent,
                    TVal const & depth)
                -> void
                {
                    extent[2u] = depth;
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The Vec<TuiDim> x offset get trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct GetOffsetX<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 1u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffsetX(
                    alpaka::Vec<TuiDim, TVal> const & extent)
                -> TVal
                {
                    return extent[0u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> x offset set trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct SetOffsetX<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 1u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto setOffsetX(
                    alpaka::Vec<TuiDim, TVal> & extent,
                    TVal const & width)
                -> void
                {
                    extent[0u] = width;
                }
            };

            //#############################################################################
            //! The Vec<TuiDim> y offset get trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct GetOffsetY<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 2u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffsetY(
                    alpaka::Vec<TuiDim, TVal> const & extent)
                -> TVal
                {
                    return extent[1u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> y offset set trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct SetOffsetY<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 2u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto setOffsetY(
                    alpaka::Vec<TuiDim, TVal> & extent,
                    TVal const & height)
                -> void
                {
                    extent[1u] = height;
                }
            };

            //#############################################################################
            //! The Vec<TuiDim> z offset get trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct GetOffsetZ<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 3u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffsetZ(
                    alpaka::Vec<TuiDim, TVal> const & extent)
                -> TVal
                {
                    return extent[2u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> z offset set trait specialization.
            //#############################################################################
            template<
                UInt TuiDim,
                typename TVal>
            struct SetOffsetZ<
                alpaka::Vec<TuiDim, TVal>,
                typename std::enable_if<(TuiDim >= 3u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto setOffsetZ(
                    alpaka::Vec<TuiDim, TVal> & extent,
                    TVal const & depth)
                -> void
                {
                    extent[2u] = depth;
                }
            };
        }
    }
}
