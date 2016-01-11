/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>      // alpaka::test::acc::TestAccs

#include <boost/test/unit_test.hpp>

#include <iostream>                     // std::cout

BOOST_AUTO_TEST_SUITE(acc)

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    getAccName,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    std::cout << alpaka::acc::getAccName<TAcc>() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
