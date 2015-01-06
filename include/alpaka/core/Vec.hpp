/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License for more details. as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC, ALPAKA_ALIGN

#include <cstdint>                  // std::uint32_t
#include <ostream>                  // std::ostream
#include <cassert>                  // assert
#include <type_traits>              // std::enable_if

// workarounds
#include <boost/predef.h>

namespace alpaka
{
    //#############################################################################
    //! A n-dimensional vector.
    //#############################################################################
    template<std::size_t TuiDim, typename TValue = std::size_t>
    class vec
    {

    public:
        static_assert(TuiDim>0, "Size of the vector is required to be greater then zero!");

        static const std::size_t s_uiDim = TuiDim;
        using ValueType = TValue;

    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //! Every value is set to zero.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC vec()
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
        template <typename TFirstArg, typename... TArgs, typename = typename std::enable_if<sizeof...(TArgs) == (TuiDim-1)>::type>
        ALPAKA_FCT_HOST_ACC vec(TFirstArg && val, TArgs && ... values)
#if !(BOOST_COMP_MSVC /*<= BOOST_VERSION_NUMBER(14, 0, 22310)*/)   // MSVC does not compile the basic array initialization: "error C2536: 'alpaka::vec<0x03>::alpaka::vec<0x03>::m_auiData': cannot specify explicit initializer for arrays"
            :
            m_auiData{std::forward<TFirstArg>(val), std::forward<TArgs>(values)...}
#endif
        {
#if BOOST_COMP_MSVC //<= BOOST_VERSION_NUMBER(14, 0, 22310)
            TValue auiData2[TuiDim] = {std::forward<TFirstArg>(val), std::forward<TArgs>(values)...};
            for(std::size_t i(0); i<TuiDim; ++i)
            {
                m_auiData[i] = auiData2[i];
            }
#endif
        }
        //-----------------------------------------------------------------------------
        //! Copy-constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC vec(vec const &) = default;
        //-----------------------------------------------------------------------------
        //! Move-constructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC vec(vec &&) = default;
        //-----------------------------------------------------------------------------
        //! Copy-assignment.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC vec & operator=(vec const &) = default;
        //-----------------------------------------------------------------------------
        //! Destructor.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC ~vec() noexcept = default;

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
        template<std::size_t TuiSubDim>
        ALPAKA_FCT_HOST_ACC vec<TuiSubDim, TValue> subvec() const
        {
            static_assert(TuiSubDim <= TuiDim, "Can not create a subvector larger then the origin vector.");

            vec<TuiSubDim, TValue> ret;

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
        ALPAKA_FCT_HOST_ACC TValue & operator[](std::size_t const uiIndex)
        {
            assert(uiIndex<TuiDim);
            return m_auiData[uiIndex];
        }
        //-----------------------------------------------------------------------------
        //! \return The value at the given index.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC TValue operator[](std::size_t const uiIndex) const
        {
            assert(uiIndex<TuiDim);
            return m_auiData[uiIndex];
        }

        //-----------------------------------------------------------------------------
        // Equality comparison operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC bool operator==(vec const & rhs) const
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
        ALPAKA_FCT_HOST_ACC bool operator!=(vec const & rhs) const
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
        /*ALPAKA_FCT_HOST_ACC static TValue dotProduct(vec const & p, vec const & q)
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
    template<std::size_t TuiDim, typename TValue>
    ALPAKA_FCT_HOST_ACC vec<TuiDim, TValue> operator+(vec<TuiDim, TValue> const & p, vec<TuiDim, TValue> const & q)
    {
        vec<TuiDim, TValue> vRet;
        for(std::size_t i(0); i<TuiDim; ++i)
        {
            vRet[i] = p[i] + q[i];
        }
        return vRet;
    }

    //-----------------------------------------------------------------------------
    //! \return The element wise product of two vectors.
    //-----------------------------------------------------------------------------
    template<std::size_t TuiDim, typename TValue>
    ALPAKA_FCT_HOST_ACC vec<TuiDim, TValue> operator*(vec<TuiDim, TValue> const & p, vec<TuiDim, TValue> const & q)
    {
        vec<TuiDim, TValue> vRet;
        for(std::size_t i(0); i<TuiDim; ++i)
        {
            vRet[i] = p[i] * q[i];
        }
        return vRet;
    }

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<std::size_t TuiDim, typename TValue>
    ALPAKA_FCT_HOST std::ostream & operator << (std::ostream & os, vec<TuiDim, TValue> const & v)
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
}
