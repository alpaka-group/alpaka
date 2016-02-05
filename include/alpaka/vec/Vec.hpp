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
#include <alpaka/dim/DimIntegralConst.hpp>  // dim::DimInt<N>
#include <alpaka/extent/Traits.hpp>         // extent::getWidth, ...
#include <alpaka/offset/Traits.hpp>         // offset::getOffsetX, ...
#include <alpaka/size/Traits.hpp>           // size::traits::SizeType

#include <alpaka/core/Align.hpp>            // ALPAKA_OPTIMAL_ALIGNMENT_SIZE
#include <alpaka/meta/IntegerSequence.hpp>  // meta::MakeIntegerSequence
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC
#include <alpaka/meta/Fold.hpp>             // meta::foldr

#include <boost/predef.h>                   // workarounds
#if !defined(__CUDA_ARCH__)
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif

#include <cstdint>                          // std::uint32_t
#include <ostream>                          // std::ostream
#include <cassert>                          // assert
#include <type_traits>                      // std::enable_if, std::decay
#include <algorithm>                        // std::min, std::max, std::min_element, std::max_element

// The nvcc compiler does not support the out of class version.
#if defined(__CUDACC__) && !defined(__CUDA__)
    #define ALPAKA_CREATE_VEC_IN_CLASS
#endif

namespace alpaka
{
    template<
        typename TDim,
        typename TSize>
    class Vec;

#ifndef ALPAKA_CREATE_VEC_IN_CLASS
    //-----------------------------------------------------------------------------
    //! Single value constructor helper.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        template<std::size_t> class TTFnObj,
        typename... TArgs,
        typename TIdxSize,
        TIdxSize... TIndices>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnArbitrary(
        meta::IntegerSequence<TIdxSize, TIndices...> const & indices,
        TArgs && ... args)
    -> Vec<TDim, decltype(TTFnObj<0>::create(std::forward<TArgs>(args)...))>
    {
#if !defined(__CUDA_ARCH__)
        boost::ignore_unused(indices);
#endif
        return Vec<TDim, decltype(TTFnObj<0>::create(std::forward<TArgs>(args)...))>(
            (TTFnObj<TIndices>::create(std::forward<TArgs>(args)...))...);
    }
    //-----------------------------------------------------------------------------
    //! Creator using func<idx>(args...) to initialize all values of the vector.
    //! The idx is in the range [0, TDim].
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        template<std::size_t> class TTFnObj,
        typename... TArgs>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFn(
        TArgs && ... args)
    -> decltype(
        createVecFromIndexedFnArbitrary<
            TDim,
            TTFnObj>(
                meta::MakeIntegerSequence<typename TDim::value_type, TDim::value>(),
                std::forward<TArgs>(args)...))
    {
        using IdxSequence = meta::MakeIntegerSequence<typename TDim::value_type, TDim::value>;
        return
            createVecFromIndexedFnArbitrary<
                TDim,
                TTFnObj>(
                    IdxSequence(),
                    std::forward<TArgs>(args)...);
    }
    //-----------------------------------------------------------------------------
    //! Creator using func<idx>(args...) to initialize all values of the vector.
    //! The idx is in the range [TIdxOffset, TIdxOffset + TDim].
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        template<std::size_t> class TTFnObj,
        typename TIdxOffset,
        typename... TArgs>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnOffset(
        TArgs && ... args)
    -> decltype(
        createVecFromIndexedFnArbitrary<
            TDim,
            TTFnObj>(
                meta::ConvertIntegerSequence<typename TIdxOffset::value_type, meta::MakeIntegerSequenceOffset<std::intmax_t, TIdxOffset::value, TDim::value>>(),
                std::forward<TArgs>(args)...))
    {
        using IdxSubSequenceSigned = meta::MakeIntegerSequenceOffset<std::intmax_t, TIdxOffset::value, TDim::value>;
        using IdxSubSequence = meta::ConvertIntegerSequence<typename TIdxOffset::value_type, IdxSubSequenceSigned>;
        return
            createVecFromIndexedFnArbitrary<
                TDim,
                TTFnObj>(
                    IdxSubSequence(),
                    std::forward<TArgs>(args)...);
    }
#endif

    //#############################################################################
    //! A n-dimensional vector.
    //#############################################################################
    template<
        typename TDim,
        typename TSize>
    class Vec final
    {
    public:
        static_assert(TDim::value>0, "Dimensionality of the vector is required to be greater then zero!");

        using Dim = TDim;
        static constexpr auto s_uiDim = TDim::value;
        using Val = TSize;

    private:
        //! A sequence of integers from 0 to dim-1.
        //! This can be used to write compile time indexing algorithms.
        using IdxSequence = meta::MakeIntegerSequence<std::size_t, TDim::value>;

    public:
        //-----------------------------------------------------------------------------
        // NOTE: No default constructor!
        //-----------------------------------------------------------------------------

        //-----------------------------------------------------------------------------
        //! Value constructor.
        //! This constructor is only available if the number of parameters matches the vector size.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TArg0,
            typename... TArgs,
            typename = typename std::enable_if<
                // There have to be dim arguments.
                (sizeof...(TArgs)+1 == TDim::value)
                &&
                (std::is_same<TSize, typename std::decay<TArg0>::type>::value)
                >::type>
        ALPAKA_FN_HOST_ACC Vec(
            TArg0 && arg0,
            TArgs && ... args) :
                m_data{std::forward<TArg0>(arg0), std::forward<TArgs>(args)...}
        {}

#ifdef ALPAKA_CREATE_VEC_IN_CLASS
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            template<std::size_t> class TTFnObj,
            typename... TArgs,
            typename TIdxSize,
            TIdxSize... TIndices>
        ALPAKA_FN_HOST_ACC static auto createVecFromIndexedFnArbitrary(
            meta::IntegerSequence<TIdxSize, TIndices...> const & indices,
            TArgs && ... args)
        -> Vec<TDim, TSize>
        {
#if !defined(__CUDA_ARCH__)
            boost::ignore_unused(indices);
#endif
            return Vec<TDim, TSize>(
                (TTFnObj<TIndices>::create(std::forward<TArgs>(args)...))...);
        }
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //! The idx is in the range [0, TDim].
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            template<std::size_t> class TTFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC static auto createVecFromIndexedFn(
            TArgs && ... args)
        -> Vec<TDim, TSize>
        {
            return
                createVecFromIndexedFnArbitrary<
                    TTFnObj>(
                        IdxSequence(),
                        std::forward<TArgs>(args)...);
        }
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //! The idx is in the range [TIdxOffset, TIdxOffset + TDim].
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            template<std::size_t> class TTFnObj,
            typename TIdxOffset,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC static auto createVecFromIndexedFnOffset(
            TArgs && ... args)
        -> Vec<TDim, TSize>
        {
            using IdxSubSequenceSigned = meta::MakeIntegerSequenceOffset<std::intmax_t, TIdxOffset::value, TDim::value>;
            using IdxSubSequence = meta::ConvertIntegerSequence<typename TDim::value_type, IdxSubSequenceSigned>;
            return
                createVecFromIndexedFnArbitrary<
                    TTFnObj>(
                        IdxSubSequence(),
                        std::forward<TArgs>(args)...);
        }
#endif

        //-----------------------------------------------------------------------------
        //! Copy constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC Vec(Vec const &) = default;
        //-----------------------------------------------------------------------------
        //! Move constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC Vec(Vec &&) = default;
        //-----------------------------------------------------------------------------
        //! Copy assignment operator.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto operator=(Vec const &) -> Vec & = default;
        //-----------------------------------------------------------------------------
        //! Move assignment operator.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto operator=(Vec &&) -> Vec & = default;
        //-----------------------------------------------------------------------------
        //! Destructor.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC ~Vec() = default;

    private:
        //#############################################################################
        //! A function object that returns the given value for each index.
        //#############################################################################
        template<
            std::size_t Tidx>
        struct CreateSingleVal
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto create(
                TSize const & val)
            -> TSize
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
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto all(
            TSize const & val)
        -> Vec<TDim, TSize>
        {
            return
                createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    TDim,
#endif
                    CreateSingleVal>(
                        val);
        }
        //-----------------------------------------------------------------------------
        //! Zero value constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto zeros()
        -> Vec<TDim, TSize>
        {
            return all(static_cast<TSize>(0));
        }
        //-----------------------------------------------------------------------------
        //! One value constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto ones()
            -> Vec<TDim, TSize>
        {
            return all(static_cast<TSize>(1));
        }

        //-----------------------------------------------------------------------------
        //! Value reference accessor at the given non-unsigned integer index.
        //! \return A reference to the value at the given index.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIdx,
            typename = typename std::enable_if<
                std::is_integral<TIdx>::value>::type>
        ALPAKA_FN_HOST_ACC auto operator[](
            TIdx const iIdx)
        -> TSize &
        {
            core::assertValueUnsigned(iIdx);
            auto const idx(static_cast<typename TDim::value_type>(iIdx));
            assert(idx<TDim::value);
            return m_data[idx];
        }

        //-----------------------------------------------------------------------------
        //! Value accessor at the given non-unsigned integer index.
        //! \return The value at the given index.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIdx,
            typename = typename std::enable_if<
                std::is_integral<TIdx>::value>::type>
        ALPAKA_FN_HOST_ACC auto operator[](
            TIdx const iIdx) const
        -> TSize
        {
            core::assertValueUnsigned(iIdx);
            auto const idx(static_cast<typename TDim::value_type>(iIdx));
            assert(idx<TDim::value);
            return m_data[idx];
        }

        //-----------------------------------------------------------------------------
        // Equality comparison operator.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto operator==(
            Vec const & rhs) const
        -> bool
        {
            for(typename TDim::value_type i(0); i < TDim::value; i++)
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
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto operator!=(
            Vec const & rhs) const
        -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        //!
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TFnObj,
            std::size_t... TIndices>
        ALPAKA_FN_HOST_ACC auto foldrByIndices(
            TFnObj const & f,
            meta::IntegerSequence<std::size_t, TIndices...> const & indices) const
        -> decltype(
            meta::foldr(
                f,
                ((*this)[TIndices])...))
        {
#if !defined(__CUDA_ARCH__)
            boost::ignore_unused(indices);
#endif
            return
                meta::foldr(
                    f,
                    ((*this)[TIndices])...);
        }
        //-----------------------------------------------------------------------------
        //!
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TFnObj>
        ALPAKA_FN_HOST_ACC auto foldrAll(
            TFnObj const & f) const
        -> decltype(
#if (BOOST_COMP_GNUC && (BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(5, 0, 0))) || __INTEL_COMPILER
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
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto prod() const
        -> TSize
        {
            return foldrAll(
                [](TSize a, TSize b)
                {
                    return a * b;
                });
        }
        //-----------------------------------------------------------------------------
        //! \return The sum of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto sum() const
        -> TSize
        {
            return foldrAll(
                [](TSize a, TSize b)
                {
                    return a + b;
                });
        }
        //-----------------------------------------------------------------------------
        //! \return The min of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto min() const
        -> TSize
        {
            return foldrAll(
                [](TSize a, TSize b)
                {
                    return (b < a) ? b : a;
                });
        }
        //-----------------------------------------------------------------------------
        //! \return The max of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto max() const
        -> TSize
        {
            return foldrAll(
                [](TSize a, TSize b)
                {
                    return (b > a) ? b : a;
                });
        }
        //-----------------------------------------------------------------------------
        //! \return The index of the minimal element.
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto minElem() const
        -> typename TDim::value_type
        {
            return
                static_cast<typename TDim::value_type>(
                    std::distance(
                        std::begin(m_data),
                        std::min_element(
                            std::begin(m_data),
                            std::end(m_data))));
        }
        //-----------------------------------------------------------------------------
        //! \return The index of the maximal element.
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto maxElem() const
        -> typename TDim::value_type
        {
            return
                static_cast<typename TDim::value_type>(
                    std::distance(
                        std::begin(m_data),
                        std::max_element(
                            std::begin(m_data),
                            std::end(m_data))));
        }

    private:
        TSize m_data[TDim::value];
    };


    namespace detail
    {
        //#############################################################################
        //! A function object that returns the sum of the two input vectors elements.
        //#############################################################################
        template<
            std::size_t Tidx>
        struct CreateAdd
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TDim,
                typename TSize>
            ALPAKA_FN_HOST_ACC static auto create(
                Vec<TDim, TSize> const & p,
                Vec<TDim, TSize> const & q)
            -> TSize
            {
                return p[Tidx] + q[Tidx];
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! \return The element wise sum of two vectors.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TSize>
    ALPAKA_FN_HOST_ACC auto operator+(
        Vec<TDim, TSize> const & p,
        Vec<TDim, TSize> const & q)
    -> Vec<TDim, TSize>
    {
        return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<TDim, TSize>::template
#endif
            createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                TDim,
#endif
                detail::CreateAdd>(
                    p,
                    q);
    }

    namespace detail
    {
        //#############################################################################
        //! A function object that returns the product of the two input vectors elements.
        //#############################################################################
        template<
            std::size_t Tidx>
        struct CreateMul
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TDim,
                typename TSize>
            ALPAKA_FN_HOST_ACC static auto create(
                Vec<TDim, TSize> const & p,
                Vec<TDim, TSize> const & q)
            -> TSize
            {
                return p[Tidx] * q[Tidx];
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! \return The element wise product of two vectors.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TSize>
    ALPAKA_FN_HOST_ACC auto operator*(
        Vec<TDim, TSize> const & p,
        Vec<TDim, TSize> const & q)
    -> Vec<TDim, TSize>
    {
        return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<TDim, TSize>::template
#endif
            createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                TDim,
#endif
                detail::CreateMul>(
                    p,
                    q);
    }

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<
        typename TDim,
        typename TSize>
    ALPAKA_FN_HOST auto operator<<(
        std::ostream & os,
        Vec<TDim, TSize> const & v)
    -> std::ostream &
    {
        os << "(";
        for(typename TDim::value_type i(0); i<TDim::value; ++i)
        {
            os << v[i];
            if(i != TDim::value-1)
            {
                os << ", ";
            }
        }
        os << ")";

        return os;
    }

    namespace vec
    {
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
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TSize,
                    std::size_t... TIndices>
                ALPAKA_FN_HOST_ACC static auto subVecFromIndices(
                    Vec<TDim, TSize> const & vec,
                    meta::IntegerSequence<std::size_t, TIndices...> const &)
                -> Vec<dim::DimInt<sizeof...(TIndices)>, TSize>
                {
                    static_assert(sizeof...(TIndices) <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");

                    return Vec<dim::DimInt<sizeof...(TIndices)>, TSize>(vec[TIndices]...);
                }
            };
            //#############################################################################
            //! Specialization for selecting the whole vector.
            //#############################################################################
            template<
                typename TDim>
            struct SubVecFromIndices<
                TDim,
                meta::MakeIntegerSequence<std::size_t, TDim::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto subVecFromIndices(
                    Vec<TDim, TSize> const & vec,
                    meta::MakeIntegerSequence<std::size_t, TDim::value> const &)
                -> Vec<TDim, TSize>
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
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TSize,
            std::size_t... TIndices>
        ALPAKA_FN_HOST_ACC auto subVecFromIndices(
            Vec<TDim, TSize> const & vec,
            meta::IntegerSequence<std::size_t, TIndices...> const & indices)
        -> Vec<dim::DimInt<sizeof...(TIndices)>, TSize>
        {
            return
                detail::SubVecFromIndices<
                    TDim,
                    meta::IntegerSequence<std::size_t, TIndices...>>
                ::subVecFromIndices(
                    vec,
                    indices);
        }
        //-----------------------------------------------------------------------------
        //! \return The sub-vector consisting of the first N elements of the source vector.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TSubDim,
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST_ACC auto subVecBegin(
            Vec<TDim, TSize> const & vec)
        -> Vec<TSubDim, TSize>
        {
            static_assert(TSubDim::value <= TDim::value, "The sub-Vec has to be smaller (or same size) then the original Vec.");

            //! A sequence of integers from 0 to dim-1.
            using IdxSubSequence = meta::MakeIntegerSequence<std::size_t, TSubDim::value>;
            return subVecFromIndices(vec, IdxSubSequence());
        }
        //-----------------------------------------------------------------------------
        //! \return The sub-vector consisting of the last N elements of the source vector.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TSubDim,
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST_ACC auto subVecEnd(
            Vec<TDim, TSize> const & vec)
        -> Vec<TSubDim, TSize>
        {
            static_assert(TSubDim::value <= TDim::value, "The sub-Vec has to be smaller (or same size) then the original Vec.");

            //! A sequence of integers from 0 to dim-1.
            using IdxSubSequence = meta::MakeIntegerSequenceOffset<std::size_t, TDim::value-TSubDim::value, TSubDim::value>;
            return subVecFromIndices(vec, IdxSubSequence());
        }

        namespace detail
        {
            //#############################################################################
            //! A function object that returns the given value for each index.
            //#############################################################################
            template<
                std::size_t Tidx>
            struct CreateCast
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TSizeNew,
                    typename TDim,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    TSizeNew const &/* valNew*/,
                    Vec<TDim, TSize> const & vec)
                -> TSizeNew
                {
                    return static_cast<TSizeNew>(vec[Tidx]);
                }
            };

    }
        //-----------------------------------------------------------------------------
        //! Cast constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TSizeNew,
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST_ACC static auto cast(Vec<TDim, TSize> const & other)
        -> Vec<TDim, TSizeNew>
        {
            return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<TDim, TSizeNew>::template
#endif
                createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    TDim,
#endif
                    detail::CreateCast>(
                        TSizeNew(),
                        other);
        }
    }
    namespace detail
    {
        //#############################################################################
        //! A function object that returns the value at the index from the back of the vector.
        //#############################################################################
        template<
            std::size_t Tidx>
        struct CreateReverse
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TDim,
                typename TSize>
            ALPAKA_FN_HOST_ACC static auto create(
                Vec<TDim, TSize> const & vec)
            -> TSize
            {
                return vec[TDim::value - 1u - Tidx];
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! \return The reverse vector.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TSize>
    ALPAKA_FN_HOST_ACC auto reverseVec(
        Vec<TDim, TSize> const & vec)
    -> Vec<TDim, TSize>
    {
        return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<TDim, TSize>::template
#endif
            createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                TDim,
#endif
                detail::CreateReverse>(
                    vec);
    }

    namespace extent
    {
        namespace detail
        {
            //#############################################################################
            //! A function object that returns the extent for each index.
            //#############################################################################
            template<
                std::size_t Tidx>
            struct CreateExtent
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent>
                ALPAKA_FN_HOST_ACC static auto create(
                    TExtent const & extent)
                -> size::Size<TExtent>
                {
                    return extent::getExtent<Tidx>(extent);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \return The extent vector.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtentVec(
            TExtent const & extent = TExtent())
        -> Vec<dim::Dim<TExtent>, size::Size<TExtent>>
        {
            return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<dim::Dim<TExtent>, size::Size<TExtent>>::template
#endif
                createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    dim::Dim<TExtent>,
#endif
                    detail::CreateExtent>(
                        extent);
        }
        //-----------------------------------------------------------------------------
        //! \return The extent but only the last N elements.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtentVecEnd(
            TExtent const & extent = TExtent())
        -> Vec<TDim, size::Size<TExtent>>
        {
            using IdxOffset = std::integral_constant<std::intmax_t, ((std::intmax_t)dim::Dim<TExtent>::value)-((std::intmax_t)TDim::value)>;
            return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<TDim, size::Size<TExtent>>::template
#endif
                createVecFromIndexedFnOffset<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    TDim,
#endif
                    detail::CreateExtent,
                    IdxOffset>(
                        extent);
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
                std::size_t Tidx>
            struct CreateOffset
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TOffsets>
                ALPAKA_FN_HOST_ACC static auto create(
                    TOffsets const & offsets)
                -> size::Size<TOffsets>
                {
                    return offset::getOffset<Tidx>(offsets);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \return The offset vector.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOffsets>
        ALPAKA_FN_HOST_ACC auto getOffsetVec(
            TOffsets const & offsets = TOffsets())
        -> Vec<dim::Dim<TOffsets>, size::Size<TOffsets>>
        {
            return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<dim::Dim<TOffsets>, size::Size<TOffsets>>::template
#endif
                createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    dim::Dim<TOffsets>,
#endif
                    detail::CreateOffset>(
                        offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The offset vector but only the last N elements.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TOffsets>
        ALPAKA_FN_HOST_ACC auto getOffsetVecEnd(
            TOffsets const & offsets = TOffsets())
        -> Vec<TDim, size::Size<TOffsets>>
        {
            using IdxOffset = std::integral_constant<std::size_t, (std::size_t)(((std::intmax_t)dim::Dim<TOffsets>::value)-((std::intmax_t)TDim::value))>;
            return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            Vec<TDim, size::Size<TOffsets>>::template
#endif
                createVecFromIndexedFnOffset<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    TDim,
#endif
                    detail::CreateOffset,
                    IdxOffset>(
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
                typename TSize>
            struct DimType<
                Vec<TDim, TSize>>
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
                typename TSize>
            struct GetExtent<
                TIdx,
                Vec<TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    Vec<TDim, TSize> const & extent)
                -> TSize
                {
                    return extent[TIdx::value];
                }
            };
            //#############################################################################
            //! The Vec extent set trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDim,
                typename TSize,
                typename TExtentVal>
            struct SetExtent<
                TIdx,
                Vec<TDim, TSize>,
                TExtentVal,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    Vec<TDim, TSize> & extent,
                    TExtentVal const & extentVal)
                -> void
                {
                    extent[TIdx::value] = extentVal;
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
                typename TSize>
            struct GetOffset<
                TIdx,
                Vec<TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    Vec<TDim, TSize> const & offsets)
                -> TSize
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
                typename TSize,
                typename TOffset>
            struct SetOffset<
                TIdx,
                Vec<TDim, TSize>,
                TOffset,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    Vec<TDim, TSize> & offsets,
                    TOffset const & offset)
                -> void
                {
                    offsets[TIdx::value] = offset;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                Vec<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
