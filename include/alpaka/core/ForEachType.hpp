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

#include <boost/mpl/is_sequence.hpp>	// boost::mpl::is_sequence
#include <boost/mpl/begin_end.hpp>		// boost::mpl::begin/end
#include <boost/mpl/deref.hpp>			// boost::mpl::deref
#include <boost/mpl/next.hpp>			// boost::mpl::next
#include <boost/mpl/aux_/unwrap.hpp>	// boost::mpl::aux::unwrap

#include <boost/predef.h>				// Workarounds.

#include <type_traits>					// std::is_same

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
		//!
        //#############################################################################
		template<
			bool TbDone = true>
		struct ForEachTypeImpl
		{
			//-----------------------------------------------------------------------------
			//!
			//-----------------------------------------------------------------------------
			template<
				typename TElemIt, 
				typename TLastIt, 
				typename TFunctor, 
				typename... TArgs>
			static void forEachTypeImpl(
				TFunctor &&,
				TArgs && ...)
			{}
		};
		
        //#############################################################################
		//!
        //#############################################################################
		template<>
		struct ForEachTypeImpl<
			false>
		{
			//-----------------------------------------------------------------------------
			//!
			//-----------------------------------------------------------------------------
			template<
				typename TElemIt, 
				typename TLastIt, 
				typename TFunctor, 
				typename... TArgs>
			static void forEachTypeImpl(
				TFunctor && f,
				TArgs && ... args)
			{
				using Elem = typename boost::mpl::deref<TElemIt>::type;

				// Call the functor template call operator.
#if BOOST_COMP_MSVC
				boost::mpl::aux::unwrap(f, 0).operator()<Elem>(std::forward<TArgs>(args)...);
#else
				boost::mpl::aux::unwrap(f, 0).template operator()<Elem>(std::forward<TArgs>(args)...);
#endif
				// Recurse to the next element.
				using NextIt = typename boost::mpl::next<TElemIt>::type;
				ForEachTypeImpl<std::is_same<NextIt, TLastIt>::value>::template forEachTypeImpl<NextIt, TLastIt>(
					std::forward<TFunctor>(f),
					std::forward<TArgs>(args)...);
			}
		};
	}

	//-----------------------------------------------------------------------------
	//! Equivalent to boost::mpl::for_each but does not require the types of the sequence to be default constructible.
	//! This function does not create instances of the types instead it passes the types as template parameter.
	//-----------------------------------------------------------------------------
	template<
		typename TSequence, 
		typename TFunctor, 
		typename... TArgs>
	void ForEachType(
		TFunctor && f,
		TArgs && ... args)
	{
		static_assert(
			boost::mpl::is_sequence<TSequence>::value, 
			"for_each_type requires the TSequence to satisfy boost::mpl::is_sequence!");

		using FirstIt = typename boost::mpl::begin<TSequence>::type;
		using LastIt = typename boost::mpl::end<TSequence>::type;

		detail::ForEachTypeImpl<std::is_same<FirstIt, LastIt>::value>::template forEachTypeImpl<FirstIt, LastIt>(
			std::forward<TFunctor>(f),
			std::forward<TArgs>(args)...);
	}
}
