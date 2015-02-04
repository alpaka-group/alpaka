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

#include <alpaka/traits/Dim.hpp>    // traits::getDim
#include <alpaka/traits/Extents.hpp>// traits::getWidth, ...

#include <cstdint>                  // std::uint32_t
#include <ostream>                  // std::ostream
#include <cassert>                  // assert
#include <type_traits>              // std::enable_if

#include <boost/mpl/and.hpp>        // boost::mpl::and_
//#include <boost/type_traits/is_convertible.hpp>

// workarounds
#include <boost/predef.h>

namespace alpaka
{
    //#############################################################################
    //! A n-dimensional vector.
    //#############################################################################
    template<
        std::size_t TuiDim, 
        // NOTE: Setting the value type to std::size_t leads to invalid data on CUDA devices (at least witch VC12).
        typename TValue = std::uint32_t>
    class Vec
    {
    public:
        static_assert(TuiDim>0, "Size of the vector is required to be greater then zero!");

        static const std::size_t s_uiDim = TuiDim;
        using Value = TValue;

    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //! Every value is set to zero.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC Vec()
        {
            // NOTE: depending on the size this could be optimized with memset, intrinsics, etc. We trust in the compiler to do this.
            for(std::size_t i(0); i<TuiDim; ++i)
            {
                m_auiData[i] = 0;
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
                && std::is_convertible<typename std::decay<TFirstArg>::type, TValue>::value
                //&& boost::mpl::and_<boost::mpl::true_, boost::mpl::true_, std::is_convertible<typename std::decay<TArgs>::type, TValue>...>::value
            >::type>
        ALPAKA_FCT_HOST_ACC Vec(TFirstArg && val, TArgs && ... values)
#if !(BOOST_COMP_MSVC /*<= BOOST_VERSION_NUMBER(14, 0, 22512)*/)   // MSVC does not compile the basic array initialization: "error C2536: 'alpaka::Vec<0x03>::alpaka::Vec<0x03>::m_auiData': cannot specify explicit initializer for arrays"
            :
            m_auiData{std::forward<TFirstArg>(val), std::forward<TArgs>(values)...}
#endif
        {
#if BOOST_COMP_MSVC //<= BOOST_VERSION_NUMBER(14, 0, 22512)
            TValue auiData2[TuiDim] = {std::forward<TFirstArg>(val), std::forward<TArgs>(values)...};
            for(std::size_t i(0); i<TuiDim; ++i)
            {
                m_auiData[i] = auiData2[i];
            }
#endif
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
        ALPAKA_FCT_HOST_ACC Vec & operator=(Vec const &) = default;
        //-----------------------------------------------------------------------------
        //! Destructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC /*virtual*/ ~Vec() noexcept = default;

        //-----------------------------------------------------------------------------
        //! Destructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC int getDim() const
        {
            return TuiDim;
        }

        //-----------------------------------------------------------------------------
        //! Constructor.
        //! Every value is set to zero.
        //-----------------------------------------------------------------------------
        template<
            std::size_t TuiSubDim>
        ALPAKA_FCT_HOST_ACC Vec<TuiSubDim, TValue> subvec() const
        {
            static_assert(TuiSubDim <= TuiDim, "The sub vector has to be smaller then the origin vector.");

            Vec<TuiSubDim, TValue> ret;

            // NOTE: depending on the size this could be optimized with memset, intrinsics, etc. We trust in the compiler to do this.
            for(std::size_t i(0); i<TuiSubDim; ++i)
            {
                ret[i] = (*this)[i];
            }

            return ret;
        }

        //-----------------------------------------------------------------------------
        //! \return A reference to the value at the given index.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC TValue & operator[](
            std::size_t const uiIdx)
        {
            assert(uiIdx<TuiDim);
            return m_auiData[uiIdx];
        }
        //-----------------------------------------------------------------------------
        //! \return The value at the given index.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC TValue operator[](
            std::size_t const uiIdx) const
        {
            assert(uiIdx<TuiDim);
            return m_auiData[uiIdx];
        }

        //-----------------------------------------------------------------------------
        // Equality comparison operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC bool operator==(
            Vec const & rhs) const
        {
            for(std::size_t i(0); i < TuiDim; i++)
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
        ALPAKA_FCT_HOST_ACC bool operator!=(
            Vec const & rhs) const
        {
            return !((*this) == rhs);
        }

        //-----------------------------------------------------------------------------
        //! \return The product of all values.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC TValue prod() const
        {
            TValue uiProd(m_auiData[0]);
            for(std::size_t i(1); i<TuiDim; ++i)
            {
                uiProd *= m_auiData[i];
            }
            return uiProd;
        }

        //-----------------------------------------------------------------------------
        //! Calculates the dot product of two vectors.
        //-----------------------------------------------------------------------------
        /*ALPAKA_FCT_HOST_ACC static TValue dotProduct(
            Vec const & p,
            Vec const & q)
        {
            TValue uiProd(0);
            for(size_t i(0); i<TuiDim; ++i)
            {
                uiProd += p[i] * q[i];
            }
            return uiProd;
        }*/

    private:
        ALPAKA_ALIGN(TValue, m_auiData[TuiDim]);
    };

    //-----------------------------------------------------------------------------
    //! \return The element wise sum of two vectors.
    //-----------------------------------------------------------------------------
    template<
        std::size_t TuiDim, 
        typename TValue>
    ALPAKA_FCT_HOST_ACC Vec<TuiDim, TValue> operator+(
        Vec<TuiDim, TValue> const & p, 
        Vec<TuiDim, TValue> const & q)
    {
        Vec<TuiDim, TValue> vRet;
        for(std::size_t i(0); i<TuiDim; ++i)
        {
            vRet[i] = p[i] + q[i];
        }
        return vRet;
    }

    //-----------------------------------------------------------------------------
    //! \return The element wise product of two vectors.
    //-----------------------------------------------------------------------------
    template<
        std::size_t TuiDim, 
        typename TValue>
    ALPAKA_FCT_HOST_ACC Vec<TuiDim, TValue> operator*(
        Vec<TuiDim, TValue> const & p, 
        Vec<TuiDim, TValue> const & q)
    {
        Vec<TuiDim, TValue> vRet;
        for(std::size_t i(0); i<TuiDim; ++i)
        {
            vRet[i] = p[i] * q[i];
        }
        return vRet;
    }

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<
        std::size_t TuiDim, 
        typename TValue>
    ALPAKA_FCT_HOST std::ostream & operator<<(
        std::ostream & os, 
        Vec<TuiDim, TValue> const & v)
    {
        os << "(";
        for(std::size_t i(0); i<TuiDim; ++i)
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
                std::size_t TuiDim>
            struct GetDim<
                alpaka::Vec<TuiDim>>
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
                std::size_t TuiDim>
            struct GetWidth<
                alpaka::Vec<TuiDim>,
                typename std::enable_if<(TuiDim >= 1u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    alpaka::Vec<TuiDim> const & extent)
                {
                    return extent[0u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> width set trait specialization.
            //#############################################################################
            template<
                std::size_t TuiDim>
            struct SetWidth<
                alpaka::Vec<TuiDim>,
                typename std::enable_if<(TuiDim >= 1u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t setWidth(
                    alpaka::Vec<TuiDim> & extent,
                    std::size_t const & width)
                {
                    return extent[0u] = width;
                }
            };

            //#############################################################################
            //! The Vec<TuiDim> height get trait specialization.
            //#############################################################################
            template<
                std::size_t TuiDim>
            struct GetHeight<
                alpaka::Vec<TuiDim>,
                typename std::enable_if<(TuiDim >= 2u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getHeight(
                    alpaka::Vec<TuiDim> const & extent)
                {
                    return extent[1u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> height set trait specialization.
            //#############################################################################
            template<
                std::size_t TuiDim>
            struct SetHeight<
                alpaka::Vec<TuiDim>,
                typename std::enable_if<(TuiDim >= 2u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t setHeight(
                    alpaka::Vec<TuiDim> & extent,
                    std::size_t const & height)
                {
                    return extent[1u] = height;
                }
            };

            //#############################################################################
            //! The Vec<TuiDim> depth get trait specialization.
            //#############################################################################
            template<
                std::size_t TuiDim>
            struct GetDepth<
                alpaka::Vec<TuiDim>,
                typename std::enable_if<(TuiDim >= 3u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getDepth(
                    alpaka::Vec<TuiDim> const & extent)
                {
                    return extent[2u];
                }
            };
            //#############################################################################
            //! The Vec<TuiDim> depth set trait specialization.
            //#############################################################################
            template<
                std::size_t TuiDim>
            struct SetDepth<
                alpaka::Vec<TuiDim>,
                typename std::enable_if<(TuiDim >= 3u) && (TuiDim <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t setDepth(
                    alpaka::Vec<TuiDim> & extent,
                    std::size_t const & depth)
                {
                    return extent[2u] = depth;
                }
            };
        }
    }
}
