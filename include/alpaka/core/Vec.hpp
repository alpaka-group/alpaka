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

#include <alpaka/dim/Traits.hpp>            // dim::getDim
#include <alpaka/dim/DimIntegralConst.hpp>  // dim::Dim<N>
#include <alpaka/extent/Traits.hpp>         // extent::getWidth, ...
#include <alpaka/offset/Traits.hpp>         // offset::getOffsetX, ...

#include <alpaka/core/IntegerSequence.hpp>  // detail::make_integer_sequence
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_ACC
#include <alpaka/core/Fold.hpp>             // foldr

#include <boost/predef.h>                   // workarounds
#if !defined(__CUDA_ARCH__)
    #include <boost/core/ignore_unused.hpp>     // boost::ignore_unused
#endif

#include <cstdint>                          // std::uint32_t
#include <ostream>                          // std::ostream
#include <cassert>                          // assert
#include <type_traits>                      // std::enable_if
#include <algorithm>                        // std::min, std::max, std::min_element, std::max_element

namespace alpaka
{
    //#############################################################################
    //! A n-dimensional vector.
    //#############################################################################
    template<
        typename TDim,
        typename TVal = Uint>
    class alignas(16u) Vec final
    {
    public:
        static_assert(TDim::value>0, "Size of the vector is required to be greater then zero!");

        using Dim = TDim;
        static constexpr Uint s_uiDim = TDim::value;
        using Val = TVal;

    private:
        //! A sequence of integers from 0 to dim-1.
        //! This can be used to write compile time indexing algorithms.
        using IdxSequence = alpaka::detail::make_integer_sequence<Uint, TDim::value>;

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
            TArgs && ... args) :
                m_auiData{std::forward<TArg0>(arg0), std::forward<TArgs>(args)...}
        {}
        //-----------------------------------------------------------------------------
        //! Single value constructor helper.
        //-----------------------------------------------------------------------------
        template<
            template<Uint> class TTFuncObj,
            typename... TArgs,
            Uint... TIndices>
        ALPAKA_FCT_HOST_ACC static auto createFromIndexedFctArbitrary(
            detail::integer_sequence<Uint, TIndices...> const & indices,
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
#if !defined(__CUDA_ARCH__)
            boost::ignore_unused(indices);
#endif
            return Vec<TDim, TVal>(
                (TTFuncObj<TIndices>::create(std::forward<TArgs>(args)...))...);
        }
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //! The idx is in the range [0, TDim].
        //-----------------------------------------------------------------------------
        template<
            template<Uint> class TTFuncObj,
            typename... TArgs>
        ALPAKA_FCT_HOST_ACC static auto createFromIndexedFct(
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
            return createFromIndexedFctArbitrary<TTFuncObj>(
                IdxSequence(),
                std::forward<TArgs>(args)...);
        }
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //! The idx is in the range [TIdxOffset, TIdxOffset + TDim].
        //-----------------------------------------------------------------------------
        template<
            template<Uint> class TTFuncObj,
            typename TIdxOffset,
            typename... TArgs>
        ALPAKA_FCT_HOST_ACC static auto createFromIndexedFctOffset(
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
            using IdxSubSequence = alpaka::detail::make_integer_sequence_start<Uint, TIdxOffset::value, TDim::value>;
            return createFromIndexedFctArbitrary<TTFuncObj>(
                IdxSubSequence(),
                std::forward<TArgs>(args)...);
        }
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //! The idx is in the range [TSrcDim-TDim, TSrcDim-TDim + TDim].
        //-----------------------------------------------------------------------------
        template<
            template<Uint> class TTFuncObj,
            typename TSrcDim,
            typename... TArgs>
        ALPAKA_FCT_HOST_ACC static auto createFromIndexedFctEnd(
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
            using IdxOffset = std::integral_constant<Uint, (Uint)(((std::intmax_t)TSrcDim::value)-((std::intmax_t)TDim::value))>;
            return createFromIndexedFctOffset<
                TTFuncObj,
                IdxOffset>(
                    std::forward<TArgs>(args)...);
        }
    private:
        //#############################################################################
        //! A function object that returns the given value for each index.
        //#############################################################################
        template<
            Uint TuiIdx>
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
            return
                createFromIndexedFct<
                    CreateSingleVal>(
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
        //-----------------------------------------------------------------------------
        //! Move constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC Vec(Vec &&) = default;
        //-----------------------------------------------------------------------------
        //! Copy assignment operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator=(Vec const &) -> Vec & = default;
        //-----------------------------------------------------------------------------
        //! Move assignment operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto operator=(Vec &&) -> Vec & = default;
        //-----------------------------------------------------------------------------
        //! Destructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC ~Vec() = default;

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
            auto const uiIdx(static_cast<Uint>(iIdx));
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
            auto const uiIdx(static_cast<Uint>(iIdx));
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
            for(Uint i(0); i < TDim::value; i++)
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
        //!
        //-----------------------------------------------------------------------------
        template<
            typename TFuncObj,
            Uint... TIndices>
        ALPAKA_FCT_HOST auto foldrByIndices(
            TFuncObj const & f,
            alpaka::detail::integer_sequence<Uint, TIndices...> const & indices) const
        -> decltype(
            foldr(
                f,
                ((*this)[TIndices])...))
        {
#if !defined(__CUDA_ARCH__)
            boost::ignore_unused(indices);
#endif
            return
                foldr(
                    f,
                    ((*this)[TIndices])...);
        }
        //-----------------------------------------------------------------------------
        //!
        //-----------------------------------------------------------------------------
        template<
            typename TFuncObj>
        ALPAKA_FCT_HOST auto foldrAll(
            TFuncObj const & f) const
        -> decltype(
#if (BOOST_COMP_GNUC) && (BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(5, 0, 0))
            this->foldrByIndices(
#else
            foldrByIndices(
#endif
                f,
                IdxSequence()))
        {
            return
                foldrByIndices(
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
        //-----------------------------------------------------------------------------
        //! \return The min of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto min() const
        -> TVal
        {
            return foldrAll(
                [](TVal a, TVal b)
                {
                    return std::min(a,b);
                });
        }
        //-----------------------------------------------------------------------------
        //! \return The max of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto max() const
        -> TVal
        {
            return foldrAll(
                [](TVal a, TVal b)
                {
                    return std::max(a,b);
                });
        }
        //-----------------------------------------------------------------------------
        //! \return The index of the minimal element.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto minElem() const
        -> Uint
        {
            return
                static_cast<Uint>(
                    std::distance(
                        std::begin(m_auiData),
                        std::min_element(
                            std::begin(m_auiData),
                            std::end(m_auiData))));
        }
        //-----------------------------------------------------------------------------
        //! \return The index of the maximal element.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto maxElem() const
        -> Uint
        {
            return
                static_cast<Uint>(
                    std::distance(
                        std::begin(m_auiData),
                        std::max_element(
                            std::begin(m_auiData),
                            std::end(m_auiData))));
        }

        //#############################################################################
        //! A function object that returns the sum of the two input vectors elements.
        //#############################################################################
        template<
            Uint TuiIdx>
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
        //! A function object that returns the product of the two input vectors elements.
        //#############################################################################
        template<
            Uint TuiIdx>
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
        // 16 Byte alignment for usage inside of CUDA kernels.
        alignas(16u) TVal m_auiData[TDim::value];
    };

    template<
        typename TVal = Uint>
    using Vec1 = Vec<dim::Dim1, TVal>;

    template<
        typename TVal = Uint>
    using Vec2 = Vec<dim::Dim2, TVal>;

    template<
        typename TVal = Uint>
    using Vec3 = Vec<dim::Dim3, TVal>;

    template<
        typename TVal = Uint>
    using Vec4 = Vec<dim::Dim4, TVal>;

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
        ::template createFromIndexedFct<
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
        ::template createFromIndexedFct<
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
        for(Uint i(0); i<TDim::value; ++i)
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
                Uint... TIndices>
            ALPAKA_FCT_HOST_ACC static auto subVecFromIndices(
                Vec<TDim, TVal> const & vec,
                alpaka::detail::integer_sequence<Uint, TIndices...> const &)
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
            alpaka::detail::make_integer_sequence<Uint, TDim::value>>
        {
            template<
                typename TVal>
            ALPAKA_FCT_HOST_ACC static auto subVecFromIndices(
                Vec<TDim, TVal> const & vec,
                alpaka::detail::make_integer_sequence<Uint, TDim::value> const &)
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
        Uint... TIndices>
    ALPAKA_FCT_HOST_ACC static auto subVecFromIndices(
        Vec<TDim, TVal> const & vec,
        detail::integer_sequence<Uint, TIndices...> const & indices)
    -> Vec<dim::Dim<sizeof...(TIndices)>, TVal>
    {
        return
            detail::SubVecFromIndices<
                TDim,
                detail::integer_sequence<Uint, TIndices...>>
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
        static_assert(TSubDim::value <= TDim::value, "The sub-Vec has to be smaller (or same size) then the original Vec.");

        //! A sequence of integers from 0 to dim-1.
        using IdxSubSequence = alpaka::detail::make_integer_sequence<Uint, TSubDim::value>;
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
        static_assert(TSubDim::value <= TDim::value, "The sub-Vec has to be smaller (or same size) then the original Vec.");

        //! A sequence of integers from 0 to dim-1.
        using IdxSubSequence = alpaka::detail::make_integer_sequence_start<Uint, TDim::value-TSubDim::value, TSubDim::value>;
        return subVecFromIndices(vec, IdxSubSequence());
    }

    namespace extent
    {
        namespace detail
        {
            //#############################################################################
            //! A function object that returns the extent for each index.
            //#############################################################################
            template<
                Uint TuiIdx,
                typename TVal>
            struct CreateExtent
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST_ACC static auto create(
                    TExtents const & extents)
                -> TVal
                {
                    return getExtent<TuiIdx, TVal>(extents);
                }
            };
            //#############################################################################
            //! NOTE: This template type is required because the template alias CreateExtentVal can not be defined inside the calling function.
            //#############################################################################
            template<
                typename TVal,
                typename TDim>
            struct GetExtentsVec
            {
                template<
                    Uint TuiIdx>
                using CreateExtentVal = CreateExtent<TuiIdx, TVal>;

                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST_ACC static auto getExtentsVec(
                    TExtents const & extents)
                -> Vec<TDim, TVal>
                {
                    return
                        Vec<TDim, TVal>
                        ::template createFromIndexedFct<
                            CreateExtentVal>(
                                extents);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST_ACC static auto getExtentsVecEnd(
                    TExtents const & extents)
                -> Vec<TDim, TVal>
                {
                    return
                        Vec<TDim, TVal>
                        ::template createFromIndexedFctEnd<
                            CreateExtentVal,
                            dim::DimT<TExtents>>(
                                extents);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \return The extents.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtentsVec(
            TExtents const & extents = TExtents())
        -> Vec<dim::DimT<TExtents>, TVal>
        {
            return
                detail::GetExtentsVec<
                    TVal,
                    dim::DimT<TExtents>>
                ::template getExtentsVec(
                    extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The extents but only the last N elements.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtentsVecEnd(
            TExtents const & extents = TExtents())
        -> Vec<dim::Dim<TDim::value>, TVal>
        {
            return
                detail::GetExtentsVec<
                    TVal,
                    dim::Dim<TDim::value>>
                ::template getExtentsVecEnd(
                    extents);
        }
    }

    namespace offset
    {
        namespace detail
        {
            //#############################################################################
            //! A function object that returns the offsets for each index.
            //#############################################################################
            template<
                Uint TuiIdx,
                typename TVal>
            struct CreateOffset
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TOffsets>
                ALPAKA_FCT_HOST_ACC static auto create(
                    TOffsets const & offsets)
                -> TVal
                {
                    return getOffset<TuiIdx, TVal>(offsets);
                }
            };
            //#############################################################################
            //! NOTE: This template type is required because the template alias CreateOffsetsVal can not be defined inside the calling function.
            //#############################################################################
            template<
                typename TVal,
                typename TDim>
            struct GetOffsetsVec
            {
                template<
                    Uint TuiIdx>
                using CreateOffsetVal = CreateOffset<TuiIdx, TVal>;

                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TOffsets>
                ALPAKA_FCT_HOST_ACC static auto getOffsetsVec(
                    TOffsets const & offsets)
                -> Vec<TDim, TVal>
                {
                    return
                        Vec<TDim, TVal>
                        ::template createFromIndexedFct<
                            CreateOffsetVal>(
                                offsets);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TOffsets>
                ALPAKA_FCT_HOST_ACC static auto getOffsetsVecEnd(
                    TOffsets const & offsets)
                -> Vec<TDim, TVal>
                {
                    return
                        Vec<TDim, TVal>
                        ::template createFromIndexedFctEnd<
                            CreateOffsetVal,
                            dim::DimT<TOffsets>>(
                                offsets);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetsVec(
            TOffsets const & offsets = TOffsets())
        -> Vec<dim::DimT<TOffsets>, TVal>
        {
            return
                detail::GetOffsetsVec<
                    TVal,
                    dim::DimT<TOffsets>>
                ::template getOffsetsVec(
                    offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets vector but only the last N elements.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetsVecEnd(
            TOffsets const & offsets = TOffsets())
        -> Vec<dim::Dim<TDim::value>, TVal>
        {
            return
                detail::GetOffsetsVec<
                    TVal,
                    dim::Dim<TDim::value>>
                ::template getOffsetsVecEnd(
                    offsets);
        }
    }

    namespace dim
    {
        namespace traits
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
    }
    namespace extent
    {
        namespace traits
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
    }
    namespace offset
    {
        namespace traits
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
