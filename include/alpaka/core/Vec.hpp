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

#include <alpaka/traits/Dim.hpp>            // traits::getDim
#include <alpaka/traits/Extent.hpp>         // traits::getWidth, ...
#include <alpaka/traits/Offset.hpp>         // traits::getOffsetX, ...

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/IntegerSequence.hpp>  // detail::make_integer_sequence
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_ACC, ALPAKA_ALIGN
#include <alpaka/core/Fold.hpp>             // foldr

#include <boost/predef.h>                   // workarounds
#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <cstdint>                          // std::uint32_t
#include <ostream>                          // std::ostream
#include <cassert>                          // assert
#include <type_traits>                      // std::enable_if

namespace alpaka
{
    //#############################################################################
    //! A n-dimensional vector.
    //#############################################################################
    template<
        typename TDim,
        typename TVal = UInt>
    class Vec
    {
    public:
        static_assert(TDim::value>0, "Size of the vector is required to be greater then zero!");

        using Dim = TDim;
        static const UInt s_uiDim = TDim::value;
        using Val = TVal;

    private:
        //! A sequence of integers from 0 to dim-1.
        //! This can be used to write compile time indexing algorithms.
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        using IdxSequence = typename alpaka::detail::make_integer_sequence<UInt, TDim::value>::type;
#else
        using IdxSequence = alpaka::detail::make_integer_sequence<UInt, TDim::value>;
#endif

    public:
        //-----------------------------------------------------------------------------
        // NOTE: No default constructor!
        //-----------------------------------------------------------------------------

        //-----------------------------------------------------------------------------
        //! Value constructor.
        //! This constructor is only available if the number of parameters matches the vector size.
        //-----------------------------------------------------------------------------
        template<
            typename TArg0,
            typename... TArgs,
            typename = typename std::enable_if<
                // There have to be dim arguments.
                (sizeof...(TArgs)+1 == TDim::value)
                && (
                    // And there is either more than one argument ...
                    (sizeof...(TArgs) > 0u)
                    // ... or the first argument is not applicable for the copy constructor.
                    || (!std::is_same<typename std::decay<TArg0>::type, Vec<TDim, TVal>>::value))
                >::type>
        ALPAKA_FCT_HOST_ACC Vec(
            TArg0 && arg0,
            TArgs && ... args)
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 22609))   // MSVC does not compile the basic array initialization: "error C2536: 'alpaka::Vec<0x03>::alpaka::Vec<0x03>::m_auiData': cannot specify explicit initializer for arrays"
            :
                m_auiData{std::forward<TArg0>(arg0), std::forward<TArgs>(args)...}
#endif
        {
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 22609))
            TVal auiData2[TDim::value] = {std::forward<TArg0>(arg0), std::forward<TArgs>(args)...};
            for(UInt i(0); i<TDim::value; ++i)
            {
                m_auiData[i] = auiData2[i];
            }
#endif
        }
    private:
        //-----------------------------------------------------------------------------
        //! Single value constructor helper.
        //-----------------------------------------------------------------------------
        template<
            template<UInt> class TTFunctor,
            typename... TArgs,
            UInt... TIndices>
        ALPAKA_FCT_HOST_ACC static auto createHelper(
            detail::integer_sequence<UInt, TIndices...> const & indices,
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
            boost::ignore_unused(indices);
            return Vec<TDim, TVal>(
                (TTFunctor<TIndices>::create(std::forward<TArgs>(args)...))...);
        }
    public:
        //-----------------------------------------------------------------------------
        //! Creator using func(idx, args...) to initialize all values of the vector.
        //-----------------------------------------------------------------------------
        template<
            template<UInt> class TTFunctor,
            typename... TArgs>
        ALPAKA_FCT_HOST_ACC static auto create(
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
            return createHelper<TTFunctor>(
                IdxSequence(),
                std::forward<TArgs>(args)...);
        }
    private:
        //#############################################################################
        //! A functor that returns the given value for each index.
        //#############################################################################
        template<
            UInt TuiIdx>
        struct CreateSingleVal
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename TVal2>
            ALPAKA_FCT_HOST_ACC static auto create(
                TVal2 && val)
            -> TVal2
            {
                return val;
            }
        };
    public:
        //-----------------------------------------------------------------------------
        //! \brief Single value constructor.
        //!
        //! Creates a vector with all values set to val.
        //! \param val The initial value.
        //-----------------------------------------------------------------------------
        template<
            typename TVal2>
        ALPAKA_FCT_HOST_ACC static auto all(
            TVal2 && val)
        -> Vec<TDim, TVal>
        {
            return create<CreateSingleVal>(
                std::forward<TVal2>(val));
        }
        //-----------------------------------------------------------------------------
        //! Zero value constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC static auto zeros()
        -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(0));
        }
        //-----------------------------------------------------------------------------
        //! One value constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC static auto ones()
            -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(1));
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
        //! Value reference accessor at the given non-unsigned integer index.
        //! \return A reference to the value at the given index.
        //-----------------------------------------------------------------------------
        template<
            typename TIdx,
            typename = typename std::enable_if<
                std::is_integral<TIdx>::value>::type>
        ALPAKA_FCT_HOST_ACC auto operator[](
            TIdx const iIdx)
        -> TVal &
        {
            assert(0<=iIdx);
            auto const uiIdx(static_cast<UInt>(iIdx));
            assert(uiIdx<TDim::value);
            return m_auiData[uiIdx];
        }

        //-----------------------------------------------------------------------------
        //! Value accessor at the given non-unsigned integer index.
        //! \return The value at the given index.
        //-----------------------------------------------------------------------------
        template<
            typename TIdx,
            typename = typename std::enable_if<
                std::is_integral<TIdx>::value>::type>
        ALPAKA_FCT_HOST_ACC auto operator[](
            TIdx const iIdx) const
        -> TVal
        {
            assert(0<=iIdx);
            auto const uiIdx(static_cast<UInt>(iIdx));
            assert(uiIdx<TDim::value);
            return m_auiData[uiIdx];
        }

        //-----------------------------------------------------------------------------
        // Equality comparison operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator==(
            Vec const & rhs) const
        -> bool
        {
            for(UInt i(0); i < TDim::value; i++)
            {
                if((*this)[i] != rhs[i])
                {
                    return false;
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
    private:
        //-----------------------------------------------------------------------------
        //!
        //-----------------------------------------------------------------------------
        template<
            typename TFunctor,
            UInt... TIndices>
        ALPAKA_FCT_HOST auto foldrAllInternal(
            TFunctor const & f,
            alpaka::detail::integer_sequence<UInt, TIndices...> const & indices) const
        -> decltype(
            foldr(
                f,
                ((*this)[TIndices])...))
        {
            boost::ignore_unused(indices);
            return
                foldr(
                    f,
                    ((*this)[TIndices])...);
        }
    public:
        //-----------------------------------------------------------------------------
        //!
        //-----------------------------------------------------------------------------
        template<
            typename TFunctor>
        ALPAKA_FCT_HOST auto foldrAll(
            TFunctor const & f) const
        -> decltype(
#if (BOOST_COMP_GNUC) && (BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(5, 0, 0))
            this->foldrAllInternal(
#else
            foldrAllInternal(
#endif
                f,
                IdxSequence()))
        {
            return
                foldrAllInternal(
                    f,
                    IdxSequence());
        }
        //-----------------------------------------------------------------------------
        //! \return The product of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto prod() const
        -> TVal
        {
            return foldrAll(std::multiplies<TVal>());
        }
        //-----------------------------------------------------------------------------
        //! \return The sum of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto sum() const
        -> TVal
        {
            return foldrAll(std::plus<TVal>());
        }

    public: // \TODO: Make private.
        //#############################################################################
        //! A functor that returns the sum of the two input vectors elements.
        //#############################################################################
        template<
            UInt TuiIdx>
        struct CreateAdd
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static auto create(
                Vec<TDim, TVal> const & p,
                Vec<TDim, TVal> const & q)
            -> TVal
            {
                return p[TuiIdx] + q[TuiIdx];
            }
        };
        //#############################################################################
        //! A functor that returns the product of the two input vectors elements.
        //#############################################################################
        template<
            UInt TuiIdx>
        struct CreateMul
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static auto create(
                Vec<TDim, TVal> const & p,
                Vec<TDim, TVal> const & q)
            -> TVal
            {
                return p[TuiIdx] * q[TuiIdx];
            }
        };

    private:
        ALPAKA_ALIGN(TVal, m_auiData[TDim::value]);
    };

    template<
        typename TVal = UInt>
    using Vec1 = Vec<dim::Dim1, TVal>;

    template<
        typename TVal = UInt>
    using Vec2 = Vec<dim::Dim2, TVal>;

    template<
        typename TVal = UInt>
    using Vec3 = Vec<dim::Dim3, TVal>;

    //-----------------------------------------------------------------------------
    //! \return The element wise sum of two vectors.
    //-----------------------------------------------------------------------------
    template<
        typename TDim,
        typename TVal>
    ALPAKA_FCT_HOST_ACC auto operator+(
        Vec<TDim, TVal> const & p,
        Vec<TDim, TVal> const & q)
    -> Vec<TDim, TVal>
    {
        return Vec<
            TDim,
            TVal>
        ::template create<
            Vec<TDim, TVal>::template CreateAdd>(
                p,
                q);
    }

    //-----------------------------------------------------------------------------
    //! \return The element wise product of two vectors.
    //-----------------------------------------------------------------------------
    template<
        typename TDim,
        typename TVal>
    ALPAKA_FCT_HOST_ACC auto operator*(
        Vec<TDim, TVal> const & p,
        Vec<TDim, TVal> const & q)
    -> Vec<TDim, TVal>
    {
        return Vec<
            TDim,
            TVal>
        ::template create<
            Vec<TDim, TVal>::template CreateMul>(
                p,
                q);
    }

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<
        typename TDim,
        typename TVal>
    ALPAKA_FCT_HOST auto operator<<(
        std::ostream & os,
        Vec<TDim, TVal> const & v)
    -> std::ostream &
    {
        os << "[";
        for(UInt i(0); i<TDim::value; ++i)
        {
            os << v[i];
            if(i<TDim::value-1)
            {
                os << ", ";
            }
        }
        os << "]";

        return os;
    }
    namespace detail
    {
        //#############################################################################
        //! Specialization for selecting a sub-vector.
        //#############################################################################
        template<
            typename TDim,
            typename TIndexSequence>
        struct SubVecFromIndices
        {
            template<
                typename TVal,
                UInt... TIndices>
            ALPAKA_FCT_HOST_ACC static auto subVecFromIndices(
                Vec<TDim, TVal> const & vec,
                alpaka::detail::integer_sequence<UInt, TIndices...> const &)
            -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
            {
                static_assert(sizeof...(TIndices) <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");

                return Vec<dim::Dim<sizeof...(TIndices)>, TVal>(vec[TIndices]...);
            }
        };
        //#############################################################################
        //! Specialization for selecting the whole vector.
        //#############################################################################
        template<
            typename TDim>
        struct SubVecFromIndices<
            TDim,
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            typename alpaka::detail::make_integer_sequence<UInt, TDim::value>::type>
#else
            alpaka::detail::make_integer_sequence<UInt, TDim::value>>
#endif
        {
            template<
                typename TVal>
            ALPAKA_FCT_HOST_ACC static auto subVecFromIndices(
                Vec<TDim, TVal> const & vec,
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                typename alpaka::detail::make_integer_sequence<UInt, TDim::value>::type const &)
#else
                alpaka::detail::make_integer_sequence<UInt, TDim::value> const &)
#endif
            -> Vec<TDim, TVal>
            {
                return vec;
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! Builds a new vector by selecting the elements of the source vector in the given order.
    //! Repeating and swizzling elements is allowed.
    //! \return The sub-vector consisting of the elements specified by the indices.
    //-----------------------------------------------------------------------------
    template<
        typename TDim,
        typename TVal,
        UInt... TIndices>
    ALPAKA_FCT_HOST_ACC static auto subVecFromIndices(
        Vec<TDim, TVal> const & vec,
        detail::integer_sequence<UInt, TIndices...> const & indices)
    -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
    {
        return
            detail::SubVecFromIndices<
                TDim,
                detail::integer_sequence<UInt, TIndices...>>
            ::subVecFromIndices(
                vec,
                indices);
    }
    //-----------------------------------------------------------------------------
    //! \return The sub-vector consisting of the first N elements of the source vector.
    //-----------------------------------------------------------------------------
    template<
        typename TSubDim,
        typename TDim,
        typename TVal>
    ALPAKA_FCT_HOST_ACC static auto subVecBegin(
        Vec<TDim, TVal> const & vec)
    -> Vec<TSubDim, TVal>
    {
        static_assert(TSubDim::value <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");

        //! A sequence of integers from 0 to dim-1.
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        using IdxSubSequence = typename alpaka::detail::make_integer_sequence<UInt, TuiSubDim::value>::type;
#else
        using IdxSubSequence = alpaka::detail::make_integer_sequence<UInt, TSubDim::value>;
#endif
        return subVecFromIndices(vec, IdxSubSequence());
    }
    //-----------------------------------------------------------------------------
    //! \return The sub-vector consisting of the last N elements of the source vector.
    //-----------------------------------------------------------------------------
    template<
        typename TSubDim,
        typename TDim,
        typename TVal>
    ALPAKA_FCT_HOST_ACC static auto subVecEnd(
        Vec<TDim, TVal> const & vec)
    -> Vec<TSubDim, TVal>
    {
        static_assert(TSubDim::value <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");

        //! A sequence of integers from 0 to dim-1.
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        using IdxSubSequence = typename alpaka::detail::make_integer_sequence_start<UInt, TDim::value-TSubDim::value, TuiSubDim::value>::type;
#else
        using IdxSubSequence = alpaka::detail::make_integer_sequence_start<UInt, TDim::value-TSubDim::value, TSubDim::value>;
#endif
        return subVecFromIndices(vec, IdxSubSequence());
    }

    namespace extent
    {
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename TVal,
                typename TExtents,
                typename TInt,
                TInt... TIndices>
            ALPAKA_FCT_HOST static auto getExtentsVecInternal(
                TExtents const & extents,
                alpaka::detail::integer_sequence<TInt, TIndices...> const &)
            -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
            {
                return {getExtent<(UInt)TIndices, TVal>(extents)...};
            }
        }
        //-----------------------------------------------------------------------------
        //! \return The extents but only the last N elements.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtentsVecNd(
            TExtents const & extents = TExtents())
        -> Vec<dim::Dim<TDim::value>>
        {
            using DimSrc = dim::DimT<TExtents>;
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            using IdxSequence = typename alpaka::detail::make_integer_sequence_start<std::intmax_t, (((std::intmax_t)DimSrc::value)-((std::intmax_t)TDim::value)), TDim::value>::type;
#else
            using IdxSequence = alpaka::detail::make_integer_sequence_start<std::intmax_t, (((std::intmax_t)DimSrc::value)-((std::intmax_t)TDim::value)), TDim::value>;
#endif
            return detail::getExtentsVecInternal<TVal>(
                extents,
                IdxSequence());
        }
        //-----------------------------------------------------------------------------
        //! \return The extents.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtentsVec(
            TExtents const & extents = TExtents())
        -> Vec<dim::Dim<dim::DimT<TExtents>::value>, TVal>
        {
            return getExtentsVecNd<dim::DimT<TExtents>, TVal>(extents);
        }
    }

    namespace offset
    {
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename TVal,
                typename TOffsets,
                typename TInt,
                TInt... TIndices>
            ALPAKA_FCT_HOST static auto getOffsetsVecInternal(
                TOffsets const & offsets,
                alpaka::detail::integer_sequence<TInt, TIndices...> const &)
            -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
            {
                return {getOffset<(UInt)TIndices, TVal>(offsets)...};
            }
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets vector but only the last N elements.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetsVecNd(
            TOffsets const & offsets = TOffsets())
        -> Vec<dim::Dim<TDim::value>, TVal>
        {
            using DimSrc = dim::DimT<TOffsets>;
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            using IdxSequence = typename alpaka::detail::make_integer_sequence_start<std::intmax_t,  (((std::intmax_t)DimSrc::value)-((std::intmax_t)TDim::value)), TDim::value>::type;
#else
            using IdxSequence = alpaka::detail::make_integer_sequence_start<std::intmax_t, (((std::intmax_t)DimSrc::value)-((std::intmax_t)TDim::value)), TDim::value>;
#endif
            return detail::getOffsetsVecInternal<TVal>(
                offsets,
                IdxSequence());
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetsVec(
            TOffsets const & offsets = TOffsets())
        -> Vec<dim::Dim<dim::DimT<TOffsets>::value>, TVal>
        {
            return getOffsetsVecNd<dim::DimT<TOffsets>, TVal>(offsets);
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The Vec dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TVal>
            struct DimType<
                alpaka::Vec<TDim, TVal>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The Vec extent get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDim,
                typename TVal>
            struct GetExtent<
                TIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    alpaka::Vec<TDim, TVal> const & extents)
                -> TVal
                {
                    return extents[TIdx::value];
                }
            };
            //#############################################################################
            //! The Vec extent set trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDim,
                typename TVal>
            struct SetExtent<
                TIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setExtent(
                    alpaka::Vec<TDim, TVal> & extents,
                    TVal2 const & extent)
                -> void
                {
                    extents[TIdx::value] = extent;
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The Vec offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDim,
                typename TVal>
            struct GetOffset<
                TIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    alpaka::Vec<TDim, TVal> const & offsets)
                -> TVal
                {
                    return offsets[TIdx::value];
                }
            };
            //#############################################################################
            //! The Vec offset set trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDim,
                typename TVal>
            struct SetOffset<
                TIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setOffset(
                    alpaka::Vec<TDim, TVal> & offsets,
                    TVal2 const & offset)
                -> void
                {
                    offsets[TIdx::value] = offset;
                }
            };
        }
    }
}
