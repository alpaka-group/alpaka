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

#include <alpaka/core/IntegerSequence.hpp>  // core::detail::make_integer_sequence
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC
#include <alpaka/core/Fold.hpp>             // core::foldr

#include <boost/predef.h>                   // workarounds
#if !defined(__CUDA_ARCH__)
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif

#include <cstdint>                          // std::uint32_t
#include <ostream>                          // std::ostream
#include <cassert>                          // assert
#include <type_traits>                      // std::enable_if
#include <algorithm>                        // std::min, std::max, std::min_element, std::max_element

namespace alpaka
{
    template<
        typename TDim,
        typename TVal>
    class Vec;

#ifndef __CUDACC__
    //-----------------------------------------------------------------------------
    //! Single value constructor helper.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        template<std::size_t> class TTFnObj,
        typename... TArgs,
        std::size_t... TIndices>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnArbitrary(
        core::detail::integer_sequence<std::size_t, TIndices...> const & indices,
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
                alpaka::core::detail::make_integer_sequence<std::size_t, TDim::value>(),
                std::forward<TArgs>(args)...))
    {
        using IdxSequence = alpaka::core::detail::make_integer_sequence<std::size_t, TDim::value>;
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
                alpaka::core::detail::make_integer_sequence_offset<std::size_t, TIdxOffset::value, TDim::value>(),
                std::forward<TArgs>(args)...))
    {
        using IdxSubSequence = alpaka::core::detail::make_integer_sequence_offset<std::size_t, TIdxOffset::value, TDim::value>;
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
        typename TVal>
    class alignas(16u) Vec final
    {
    public:
        static_assert(TDim::value>0, "Dimensionality of the vector is required to be greater then zero!");

        using Dim = TDim;
        static constexpr auto s_uiDim = TDim::value;
        using Val = TVal;

    private:
        //! A sequence of integers from 0 to dim-1.
        //! This can be used to write compile time indexing algorithms.
        using IdxSequence = alpaka::core::detail::make_integer_sequence<std::size_t, TDim::value>;

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
                && (
                    // And there is either more than one argument ...
                    (sizeof...(TArgs) > 0u)
                    // ... or the first argument is not applicable for the copy constructor.
                    || (!std::is_same<typename std::decay<TArg0>::type, Vec<TDim, TVal>>::value))
                >::type>
        ALPAKA_FN_HOST_ACC Vec(
            TArg0 && arg0,
            TArgs && ... args) :
                m_auiData{std::forward<TArg0>(arg0), std::forward<TArgs>(args)...}
        {}

#ifdef __CUDACC__
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            template<std::size_t> class TTFnObj,
            typename... TArgs,
            std::size_t... TIndices>
        ALPAKA_FN_HOST_ACC static auto createVecFromIndexedFnArbitrary(
            core::detail::integer_sequence<std::size_t, TIndices...> const & indices,
            TArgs && ... args)
        -> Vec<TDim, TVal>
        {
#if !defined(__CUDA_ARCH__)
            boost::ignore_unused(indices);
#endif
            return Vec<TDim, TVal>(
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
        -> Vec<TDim, TVal>
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
        -> Vec<TDim, TVal>
        {
            using IdxSubSequence = alpaka::core::detail::make_integer_sequence_offset<std::size_t, TIdxOffset::value, TDim::value>;
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
            std::size_t TuiIdx>
        struct CreateSingleVal
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto create(
                TVal const & val)
            -> TVal
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
            TVal const & val)
        -> Vec<TDim, TVal>
        {
            return
                createVecFromIndexedFn<
#ifndef __CUDACC__
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
        -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(0));
        }
        //-----------------------------------------------------------------------------
        //! One value constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto ones()
            -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(1));
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
        -> TVal &
        {
            core::assertValueUnsigned(iIdx);
            auto const uiIdx(static_cast<typename TDim::value_type>(iIdx));
            assert(uiIdx<TDim::value);
            return m_auiData[uiIdx];
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
        -> TVal
        {
            core::assertValueUnsigned(iIdx);
            auto const uiIdx(static_cast<typename TDim::value_type>(iIdx));
            assert(uiIdx<TDim::value);
            return m_auiData[uiIdx];
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
            alpaka::core::detail::integer_sequence<std::size_t, TIndices...> const & indices) const
        -> decltype(
            core::foldr(
                f,
                ((*this)[TIndices])...))
        {
#if !defined(__CUDA_ARCH__)
            boost::ignore_unused(indices);
#endif
            return
                core::foldr(
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
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto prod() const
        -> TVal
        {
            return foldrAll(std::multiplies<TVal>());
        }
        //-----------------------------------------------------------------------------
        //! \return The sum of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto sum() const
        -> TVal
        {
            return foldrAll(std::plus<TVal>());
        }
        //-----------------------------------------------------------------------------
        //! \return The min of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto min() const
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
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto max() const
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
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto minElem() const
        -> typename TDim::value_type
        {
            return
                static_cast<typename TDim::value_type>(
                    std::distance(
                        std::begin(m_auiData),
                        std::min_element(
                            std::begin(m_auiData),
                            std::end(m_auiData))));
        }
        //-----------------------------------------------------------------------------
        //! \return The index of the maximal element.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto maxElem() const
        -> typename TDim::value_type
        {
            return
                static_cast<typename TDim::value_type>(
                    std::distance(
                        std::begin(m_auiData),
                        std::max_element(
                            std::begin(m_auiData),
                            std::end(m_auiData))));
        }

    private:
        // 16 Byte alignment for usage inside of CUDA kernels.
        alignas(16u) TVal m_auiData[TDim::value];
    };

    template<
        typename TVal>
    using Vec1 = Vec<dim::DimInt<1u>, TVal>;

    template<
        typename TVal>
    using Vec2 = Vec<dim::DimInt<2u>, TVal>;

    template<
        typename TVal>
    using Vec3 = Vec<dim::DimInt<3u>, TVal>;

    template<
        typename TVal>
    using Vec4 = Vec<dim::DimInt<4u>, TVal>;

    namespace detail
    {
        //#############################################################################
        //! A function object that returns the sum of the two input vectors elements.
        //#############################################################################
        template<
            std::size_t TuiIdx>
        struct CreateAdd
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TDim,
                typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(
                Vec<TDim, TVal> const & p,
                Vec<TDim, TVal> const & q)
            -> TVal
            {
                return p[TuiIdx] + q[TuiIdx];
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! \return The element wise sum of two vectors.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TVal>
    ALPAKA_FN_HOST_ACC auto operator+(
        Vec<TDim, TVal> const & p,
        Vec<TDim, TVal> const & q)
    -> Vec<TDim, TVal>
    {
        return
#ifdef __CUDACC__
            Vec<TDim, TVal>::template
#endif
            createVecFromIndexedFn<
#ifndef __CUDACC__
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
            std::size_t TuiIdx>
        struct CreateMul
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TDim,
                typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(
                Vec<TDim, TVal> const & p,
                Vec<TDim, TVal> const & q)
            -> TVal
            {
                return p[TuiIdx] * q[TuiIdx];
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! \return The element wise product of two vectors.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TVal>
    ALPAKA_FN_HOST_ACC auto operator*(
        Vec<TDim, TVal> const & p,
        Vec<TDim, TVal> const & q)
    -> Vec<TDim, TVal>
    {
        return
#ifdef __CUDACC__
            Vec<TDim, TVal>::template
#endif
            createVecFromIndexedFn<
#ifndef __CUDACC__
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
        typename TVal>
    ALPAKA_FN_HOST auto operator<<(
        std::ostream & os,
        Vec<TDim, TVal> const & v)
    -> std::ostream &
    {
        os << "(";
        for(typename TDim::value_type i(0); i<TDim::value; ++i)
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
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TVal,
                std::size_t... TIndices>
            ALPAKA_FN_HOST_ACC static auto subVecFromIndices(
                Vec<TDim, TVal> const & vec,
                alpaka::core::detail::integer_sequence<std::size_t, TIndices...> const &)
            -> Vec<dim::DimInt<sizeof...(TIndices)>, TVal>
            {
                static_assert(sizeof...(TIndices) <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");

                return Vec<dim::DimInt<sizeof...(TIndices)>, TVal>(vec[TIndices]...);
            }
        };
        //#############################################################################
        //! Specialization for selecting the whole vector.
        //#############################################################################
        template<
            typename TDim>
        struct SubVecFromIndices<
            TDim,
            alpaka::core::detail::make_integer_sequence<std::size_t, TDim::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TVal>
            ALPAKA_FN_HOST_ACC static auto subVecFromIndices(
                Vec<TDim, TVal> const & vec,
                alpaka::core::detail::make_integer_sequence<std::size_t, TDim::value> const &)
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
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TVal,
        std::size_t... TIndices>
    ALPAKA_FN_HOST_ACC auto subVecFromIndices(
        Vec<TDim, TVal> const & vec,
        core::detail::integer_sequence<std::size_t, TIndices...> const & indices)
    -> Vec<dim::DimInt<sizeof...(TIndices)>, TVal>
    {
        return
            detail::SubVecFromIndices<
                TDim,
                core::detail::integer_sequence<std::size_t, TIndices...>>
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
        typename TVal>
    ALPAKA_FN_HOST_ACC auto subVecBegin(
        Vec<TDim, TVal> const & vec)
    -> Vec<TSubDim, TVal>
    {
        static_assert(TSubDim::value <= TDim::value, "The sub-Vec has to be smaller (or same size) then the original Vec.");

        //! A sequence of integers from 0 to dim-1.
        using IdxSubSequence = alpaka::core::detail::make_integer_sequence<std::size_t, TSubDim::value>;
        return subVecFromIndices(vec, IdxSubSequence());
    }
    //-----------------------------------------------------------------------------
    //! \return The sub-vector consisting of the last N elements of the source vector.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TSubDim,
        typename TDim,
        typename TVal>
    ALPAKA_FN_HOST_ACC auto subVecEnd(
        Vec<TDim, TVal> const & vec)
    -> Vec<TSubDim, TVal>
    {
        static_assert(TSubDim::value <= TDim::value, "The sub-Vec has to be smaller (or same size) then the original Vec.");

        //! A sequence of integers from 0 to dim-1.
        using IdxSubSequence = alpaka::core::detail::make_integer_sequence_offset<std::size_t, TDim::value-TSubDim::value, TSubDim::value>;
        return subVecFromIndices(vec, IdxSubSequence());
    }

    namespace detail
    {
        //#############################################################################
        //! A function object that returns the given value for each index.
        //#############################################################################
        template<
            std::size_t TuiIdx>
        struct CreateCast
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TValNew,
                typename TDim,
                typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(
                TValNew const &/* valNew*/,
                Vec<TDim, TVal> const & vec)
            -> TValNew
            {
                return static_cast<TValNew>(vec[TuiIdx]);
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! Cast constructor.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TValNew,
        typename TDim,
        typename TVal>
    ALPAKA_FN_HOST_ACC static auto castVec(Vec<TDim, TVal> const & other)
    -> Vec<TDim, TValNew>
    {
        return
#ifdef __CUDACC__
        Vec<TDim, TValNew>::template
#endif
            createVecFromIndexedFn<
#ifndef __CUDACC__
                TDim,
#endif
                detail::CreateCast>(
                    TValNew(),
                    other);
    }
    namespace detail
    {
        //#############################################################################
        //! A function object that returns the value at the index from the back of the vector.
        //#############################################################################
        template<
            std::size_t TuiIdx>
        struct CreateReverse
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TDim,
                typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(
                Vec<TDim, TVal> const & vec)
            -> TVal
            {
                return vec[TDim::value - 1u - TuiIdx];
            }
        };
    }
    //-----------------------------------------------------------------------------
    //! \return The reverse vector.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TVal>
    ALPAKA_FN_HOST_ACC auto reverseVec(
        Vec<TDim, TVal> const & vec)
    -> Vec<TDim, TVal>
    {
        return
#ifdef __CUDACC__
            Vec<TDim, TVal>::template
#endif
            createVecFromIndexedFn<
#ifndef __CUDACC__
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
                std::size_t TuiIdx>
            struct CreateExtent
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtents>
                ALPAKA_FN_HOST_ACC static auto create(
                    TExtents const & extents)
                -> size::Size<TExtents>
                {
                    return extent::getExtent<TuiIdx>(extents);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \return The extents.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents>
        ALPAKA_FN_HOST_ACC auto getExtentsVec(
            TExtents const & extents = TExtents())
        -> Vec<dim::Dim<TExtents>, size::Size<TExtents>>
        {
            return
#ifdef __CUDACC__
            Vec<dim::Dim<TExtents>, size::Size<TExtents>>::template
#endif
                createVecFromIndexedFn<
#ifndef __CUDACC__
                    dim::Dim<TExtents>,
#endif
                    detail::CreateExtent>(
                        extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The extents but only the last N elements.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TExtents>
        ALPAKA_FN_HOST_ACC auto getExtentsVecEnd(
            TExtents const & extents = TExtents())
        -> Vec<TDim, size::Size<TExtents>>
        {
            using IdxOffset = std::integral_constant<std::size_t, (std::size_t)(((std::intmax_t)dim::Dim<TExtents>::value)-((std::intmax_t)TDim::value))>;
            return
#ifdef __CUDACC__
            Vec<TDim, size::Size<TExtents>>::template
#endif
                createVecFromIndexedFnOffset<
#ifndef __CUDACC__
                    TDim,
#endif
                    detail::CreateExtent,
                    IdxOffset>(
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
                std::size_t TuiIdx>
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
                    return offset::getOffset<TuiIdx>(offsets);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOffsets>
        ALPAKA_FN_HOST_ACC auto getOffsetsVec(
            TOffsets const & offsets = TOffsets())
        -> Vec<dim::Dim<TOffsets>, size::Size<TOffsets>>
        {
            return
#ifdef __CUDACC__
            Vec<dim::Dim<TOffsets>, size::Size<TOffsets>>::template
#endif
                createVecFromIndexedFn<
#ifndef __CUDACC__
                    dim::Dim<TOffsets>,
#endif
                    detail::CreateOffset>(
                        offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The offsets vector but only the last N elements.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TOffsets>
        ALPAKA_FN_HOST_ACC auto getOffsetsVecEnd(
            TOffsets const & offsets = TOffsets())
        -> Vec<TDim, size::Size<TOffsets>>
        {
            using IdxOffset = std::integral_constant<std::size_t, (std::size_t)(((std::intmax_t)dim::Dim<TOffsets>::value)-((std::intmax_t)TDim::value))>;
            return
#ifdef __CUDACC__
            Vec<TDim, size::Size<TOffsets>>::template
#endif
                createVecFromIndexedFnOffset<
#ifndef __CUDACC__
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
                typename TVal>
            struct DimType<
                Vec<TDim, TVal>>
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
                Vec<TDim, TVal>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    Vec<TDim, TVal> const & extents)
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
                typename TVal,
                typename TExtent>
            struct SetExtent<
                TIdx,
                Vec<TDim, TVal>,
                TExtent,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    Vec<TDim, TVal> & extents,
                    TExtent const & extent)
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
                Vec<TDim, TVal>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    Vec<TDim, TVal> const & offsets)
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
                typename TVal,
                typename TOffset>
            struct SetOffset<
                TIdx,
                Vec<TDim, TVal>,
                TOffset,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    Vec<TDim, TVal> & offsets,
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
                typename TVal>
            struct SizeType<
                Vec<TDim, TVal>>
            {
                using type = TVal;
            };
        }
    }
}
