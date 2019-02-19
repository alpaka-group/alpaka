/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include <alpaka/alpaka.hpp>
#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/clear.hpp>
#include <boost/mpl/pop_back.hpp>
#include <boost/mpl/contains.hpp>

#include <iostream>
#include <type_traits>
#include <typeinfo>

//-----------------------------------------------------------------------------
// This code is based on:
// http://stackoverflow.com/questions/5099429/how-to-use-stdtuple-types-with-boostmpl-algorithms/15865204#15865204
TEST_CASE("stdTupleAsMplSequence", "[meta]")
{
    using Tuple = std::tuple<int, char, bool>;

    static_assert(
        std::is_same<boost::mpl::front<Tuple>::type, int>::value,
        "boost::mpl::front on the std::tuple failed!");
    static_assert(
        boost::mpl::size<Tuple>::type::value == 3,
        "boost::mpl::size on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::pop_front<Tuple>::type, std::tuple<char, bool>>::value,
        "boost::mpl::pop_front on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::push_front<Tuple, unsigned>::type, std::tuple<unsigned, int, char, bool>>::value,
        "boost::mpl::push_front on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::push_back<Tuple, unsigned>::type, std::tuple<int, char, bool, unsigned>>::value,
        "boost::mpl::push_back on the std::tuple failed!");
    static_assert(
        boost::mpl::empty<Tuple>::type::value == false,
        "boost::mpl::empty on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::at_c<Tuple, 0>::type, int>::value,
        "boost::mpl::at_c on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::at_c<Tuple, 1>::type, char>::value,
        "boost::mpl::at_c on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::back<Tuple>::type, bool>::value,
        "boost::mpl::back on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::clear<Tuple>::type, std::tuple<>>::value,
        "boost::mpl::clear on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::pop_back<Tuple>::type, std::tuple<int, char>>::value,
        "boost::mpl::pop_back on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, int>::value,
        "boost::mpl::contains on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, char>::value,
        "boost::mpl::contains on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, bool>::value,
        "boost::mpl::contains on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, unsigned>::value == false,
        "boost::mpl::contains on the std::tuple failed!");
}

//-----------------------------------------------------------------------------
struct TestTemplate
{
    template< typename T >
    void operator()()
    {
        std::cout << typeid(T).name() << std::endl;
    }
};

using TestTuple = std::tuple<int, char, bool>;

TEST_CASE( "stdTupleAsMplSequenceTemplateTest", "[meta]")
{
    alpaka::meta::forEachType< TestTuple >( TestTemplate() );
}
