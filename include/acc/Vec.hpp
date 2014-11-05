/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of acc.
*
* acc is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* acc is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with acc.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <acc/FctCudaCpu.hpp>	// ACC_FCT_CPU_CUDA

#include <cstdint>				// std::uint32_t
#include <ostream>				// std::ostream
#include <cassert>				// assert
#include <type_traits>			// std::enable_if

namespace acc
{
	//#############################################################################
	//! A 3-dimensional vector.
	//#############################################################################
	template<std::size_t UiSize>
	class vec
	{
	public:
		static_assert(UiSize>0, "Size of the vector is required to be greater then zero!");

		//-----------------------------------------------------------------------------
		//! Constructor.
		//! Every value is set to zero.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA vec()
		{
			// NOTE: depending on the size this could be optimized with memeset, intrinsics, etc. We trust in the compiler to do this.
			for(std::size_t i(0); i<UiSize; ++i)
			{
				m_auiData[i] = 0;
			}
		}

		//-----------------------------------------------------------------------------
		//! Value-constructor.
		//! This constructor is only available if the number of parameters matches the vector size.
		//-----------------------------------------------------------------------------
		template <typename TFirstArg, typename... TArgs, typename = typename std::enable_if<sizeof...(TArgs) == (UiSize-1)>::type>
		ACC_FCT_CPU_CUDA vec(TFirstArg && val, TArgs && ... values)
#ifndef _MSC_VER	// MSVC <= 14 do not compile the basic array initialization.
			:
			m_auiData{std::forward<TFirstArg>(val), std::forward<TArgs>(values)...}
#endif
		{
#ifdef _MSC_VER
			std::uint32_t auiData2[UiSize] = {std::forward<TFirstArg>(val), std::forward<TArgs>(values)...};
			for(std::size_t i(0); i<UiSize; ++i)
			{
				m_auiData[i] = auiData2[i];
			}
#endif
		}

		//-----------------------------------------------------------------------------
		//! \return A reference to the value at the given index.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA std::uint32_t & operator[](std::size_t const uiIndex)
		{
			assert(uiIndex<UiSize);
			return m_auiData[uiIndex];
		}
		//-----------------------------------------------------------------------------
		//! \return The value at the given index.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA std::uint32_t operator[](std::size_t const uiIndex) const
		{
			assert(uiIndex<UiSize);
			return m_auiData[uiIndex];
		}

		//-----------------------------------------------------------------------------
		//! \return The product of all values.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA std::uint32_t prod() const
		{
			std::uint32_t uiProd(m_auiData[0]);
			for(std::size_t i(1); i<UiSize; ++i)
			{
				uiProd *= m_auiData[i];
			}
			return uiProd;
		}

		//-----------------------------------------------------------------------------
		//! Calculates the dot product of two vectors.
		//-----------------------------------------------------------------------------
		/*ACC_FCT_CPU_CUDA static std::uint32_t dotProduct(vec const & p, vec const & q)
		{
			std::uint32_t uiProd(0);
			for(size_t i(0); i<UiSize; ++i)
			{
				uiProd += p[i] * q[i];
			}
			return uiProd;
		}*/

	private:
		// NOTE: We can not use std::array here because the operator[] i a host function not available within a kernel.
		std::uint32_t m_auiData[UiSize];
	};

	//-----------------------------------------------------------------------------
	//! \return The element wise sum of two vectors.
	//-----------------------------------------------------------------------------
	template<std::size_t UiSize>
	ACC_FCT_CPU_CUDA vec<UiSize> operator+(vec<UiSize> const & p, vec<UiSize> const & q)
	{
		vec<UiSize> vRet;
		for(std::size_t i(0); i<UiSize; ++i)
		{
			vRet[i] = p[i] + q[i];
		}
		return vRet;
	}

	//-----------------------------------------------------------------------------
	//! \return The element wise product of two vectors.
	//-----------------------------------------------------------------------------
	template<std::size_t UiSize>
	ACC_FCT_CPU_CUDA vec<UiSize> operator*(vec<UiSize> const & p, vec<UiSize> const & q)
	{
		vec<UiSize> vRet;
		for(std::size_t i(0); i<UiSize; ++i)
		{
			vRet[i] = p[i] * q[i];
		}
		return vRet;
	}

	//-----------------------------------------------------------------------------
	//! Stream out operator.
	//-----------------------------------------------------------------------------
	template<std::size_t UiSize>
	ACC_FCT_CPU std::ostream & operator << (std::ostream & os, vec<UiSize> const & v)
	{
		os << "(";
		for(std::size_t i(0); i<UiSize; ++i)
		{
			os << v[i];
			if(i<UiSize-1)
			{
				os << ", ";
			}
		}
		os << ")";

		return os;
	}
}