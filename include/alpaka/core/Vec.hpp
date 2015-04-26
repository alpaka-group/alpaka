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

#include <alpaka/traits/Dim.hpp>    // traits::getDim
#include <alpaka/traits/Extent.hpp> // traits::getWidth, ...
#include <alpaka/traits/Offset.hpp> // traits::getOffsetX, ...

#include <alpaka/core/BasicDims.hpp>// dim::Dim<N>
#include <alpaka/core/IntegerSequence.hpp>    //
#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_ACC, ALPAKA_ALIGN

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
            typename... TArgs,
            typename = typename std::enable_if<sizeof...(TArgs) == TDim::value>::type>
        ALPAKA_FCT_HOST_ACC Vec(
            TArgs && ... vals)
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 22609))   // MSVC does not compile the basic array initialization: "error C2536: 'alpaka::Vec<0x03>::alpaka::Vec<0x03>::m_auiData': cannot specify explicit initializer for arrays"
            :
                m_auiData{std::forward<TArgs>(vals)...}
#endif
        {
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 22609))
            TVal auiData2[TDim::value] = {std::forward<TArgs>(vals)...};
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
            detail::integer_sequence<UInt, TIndices...> const &,
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
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
        //! Destructor.
        //-----------------------------------------------------------------------------
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        ALPAKA_FCT_HOST_ACC auto static getDim()
#else
        ALPAKA_FCT_HOST_ACC auto static constexpr getDim()
#endif
        -> UInt
        {
            return TDim::value;
        }

        //-----------------------------------------------------------------------------
        //! Value reference accessor at the given non-unsigned integer index.
        //! \return A reference to the value at the given index.
        //-----------------------------------------------------------------------------
        template<
            typename TIdx,
            typename = typename std::enable_if<
                std::is_integral<TIdx>::value
                /*&& !std::is_unsigned<TIdx>::value*/>::type/* * = nullptr*/>
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
        //! Value reference accessor at the given unsigned integer index.
        //! \return A reference to the value at the given index.
        //-----------------------------------------------------------------------------
        /*template<
            typename TIdx,
            typename std::enable_if<
                std::is_integral<TIdx>::value
                && std::is_unsigned<TIdx>::value>::type * = nullptr>
        ALPAKA_FCT_HOST_ACC auto operator[](
            TIdx const uiIdx)
        -> TVal &
        {
            assert(uiIdx<TDim::value);
            return m_auiData[uiIdx];
        }*/

        //-----------------------------------------------------------------------------
        //! Value accessor at the given non-unsigned integer index.
        //! \return The value at the given index.
        //-----------------------------------------------------------------------------
        template<
            typename TIdx,
            typename = typename std::enable_if<
                std::is_integral<TIdx>::value
                /*&& !std::is_unsigned<TIdx>::value*/>::type/* * = nullptr*/>
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
        //! Value accessor at the given unsigned integer index.
        //! \return The value at the given index.
        //-----------------------------------------------------------------------------
        /*template<
            typename TIdx,
            typename std::enable_if<
                std::is_integral<TIdx>::value
                && std::is_unsigned<TIdx>::value>::type * = nullptr>
        ALPAKA_FCT_HOST_ACC auto operator[](
            TIdx const uiIdx) const
        -> TVal
        {
            assert(uiIdx<TDim::value);
            return m_auiData[uiIdx];
        }*/

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

        //-----------------------------------------------------------------------------
        //! \return The product of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto prod() const
        -> TVal
        {
            TVal uiProd(m_auiData[0]);
            for(UInt i(1); i<TDim::value; ++i)
            {
                uiProd *= m_auiData[i];
            }
            return uiProd;
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
        os << "(";
        for(UInt i(0); i<TDim::value; ++i)
        {
            os << v[i];
            if(i<TDim::value-1)
            {
                os << ", ";
            }
        }
        os << ")";

        return os;
    }
    
    //-----------------------------------------------------------------------------
    //! \return The sub-vector consisting of the elements specified by the indices.
    //-----------------------------------------------------------------------------
    template<
        UInt... TIndices,
        typename TDim,
        typename TVal>
    ALPAKA_FCT_HOST_ACC static auto subVecFromIndices(
        Vec<TDim, TVal> const & vec,
        detail::integer_sequence<UInt, TIndices...> const &)
    -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
    {
        static_assert(sizeof...(TIndices) <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");

        return Vec<dim::Dim<sizeof...(TIndices)>, TVal>(vec[TIndices]...);
    }
    //-----------------------------------------------------------------------------
    //! \return The sub-vector consisting of the first N elements of the source vector.
    //-----------------------------------------------------------------------------
    template<
        typename TSubDim,
        typename TDim,
        typename TVal/*,
        typename std::enable_if<TuiSubDim != TDim::value>::type * = nullptr*/>
    ALPAKA_FCT_HOST_ACC static auto subVec(
        Vec<TDim, TVal> const & vec)
    -> Vec<TSubDim, TVal>
    {
        static_assert(TSubDim::value <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");
        
        //! A sequence of integers from 0 to dim-1.
        //! This can be used to write compile time indexing algorithms.
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        using IdxSubSequence = typename alpaka::detail::make_integer_sequence<UInt, TuiSubDim>::type;
#else
        using IdxSubSequence = alpaka::detail::make_integer_sequence<UInt, TSubDim::value>;
#endif
        return subVecFromIndices(vec, IdxSubSequence());
    }
    //-----------------------------------------------------------------------------
    //! \return The sub-vector consisting of the first N elements of the source vector.
    //! For subDim == dim nothing has to be done.
    // Specialization for template class template methods is not possible, SFINAE as workaround does.
    // \FIXME: Does not work with nvcc!
    //-----------------------------------------------------------------------------
    /*template<
        typename TSubDim,
        typename TDim,
        typename TVal,
        typename std::enable_if<TSubDim::value == TDim::value>::type * = nullptr>
    ALPAKA_FCT_HOST_ACC static auto subVec(
        Vec<TDim, TVal> const & vec)
    -> Vec<TSubDim, TVal>
    {
        return vec;
    }*/
    
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
                size_t... TIndices>
            ALPAKA_FCT_HOST static auto getExtentsInternal(
                TExtents const & extents,
#if !BOOST_COMP_MSVC     // MSVC 190022512 introduced a new bug with alias templates: error C3520: 'TIndices': parameter pack must be expanded in this context
            alpaka::detail::index_sequence<TIndices...> const &)
#else
            alpaka::detail::integer_sequence<std::size_t, TIndices...> const &)
#endif
            -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
            {
                return {getExtent<TIndices, TVal>(extents)...};
            }
        }
        //-----------------------------------------------------------------------------
        //! \return The extents.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtentsNd(
            TExtents const & extents = TExtents())
        -> Vec<TDim>
        {
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            using IdxSequence = typename alpaka::detail::make_index_sequence<TDim::value>::type;
#else
            using IdxSequence = alpaka::detail::make_index_sequence<TDim::value>;
#endif
            return detail::getExtentsInternal<TVal>(
                extents,
                IdxSequence());
        }
        //-----------------------------------------------------------------------------
        //! \return The extents.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtents(
            TExtents const & extents = TExtents())
        -> Vec<dim::DimT<TExtents>, TVal>
        {
            return getExtentsNd<dim::DimT<TExtents>, TVal>(extents);
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
                size_t... TIndices>
            ALPAKA_FCT_HOST static auto getOffsetsInternal(
                TOffsets const & extents,
#if !BOOST_COMP_MSVC     // MSVC 190022512 introduced a new bug with alias templates: error C3520: 'TIndices': parameter pack must be expanded in this context
            alpaka::detail::index_sequence<TIndices...> const &)
#else
            alpaka::detail::integer_sequence<std::size_t, TIndices...> const &)
#endif
            -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
            {
                return {getOffset<TIndices, TVal>(extents)...};
            }
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetsNd(
            TOffsets const & offsets = TOffsets())
        -> Vec<TDim, TVal>
        {
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            using IdxSequence = typename alpaka::detail::make_index_sequence<TDim::value>::type;
#else
            using IdxSequence = alpaka::detail::make_index_sequence<TDim::value>;
#endif
            return detail::getOffsetsInternal<TVal>(
                offsets,
                IdxSequence());
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsets(
            TOffsets const & offsets = TOffsets())
        -> Vec<dim::DimT<TOffsets>, TVal>
        {
            return getOffsetsNd<dim::DimT<TOffsets>, TVal>(offsets);
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
            //! The Vec width get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TDim,
                typename TVal>
            struct GetExtent<
                TuiIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    alpaka::Vec<TDim, TVal> const & extents)
                -> TVal
                {
                    return extents[TuiIdx];
                }
            };
            //#############################################################################
            //! The Vec width set trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TDim,
                typename TVal>
            struct SetExtent<
                TuiIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setExtent(
                    alpaka::Vec<TDim, TVal> & extents,
                    TVal2 const & extent)
                -> void
                {
                    extents[TuiIdx] = extent;
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The Vec offset get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TDim,
                typename TVal>
            struct GetOffset<
                TuiIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    alpaka::Vec<TDim, TVal> const & offsets)
                -> TVal
                {
                    return offsets[TuiIdx];
                }
            };
            //#############################################################################
            //! The Vec offset set trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TDim,
                typename TVal>
            struct SetOffset<
                TuiIdx,
                alpaka::Vec<TDim, TVal>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setOffsetX(
                    alpaka::Vec<TDim, TVal> & offsets,
                    TVal2 const & offset)
                -> void
                {
                    offsets[TuiIdx] = offset;
                }
            };
        }
    }
}
