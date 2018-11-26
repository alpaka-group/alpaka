/**
 * \file
 * Copyright 2018-2019 Benjamin Worpitz
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

#include <catch2/catch.hpp>

#include <iostream>

//-----------------------------------------------------------------------------
TEST_CASE("printDefines", "[core]")
{
#if BOOST_LANG_CUDA
    std::cout << "BOOST_LANG_CUDA:" << BOOST_LANG_CUDA << std::endl;
#endif
#if BOOST_LANG_HIP
    std::cout << "BOOST_LANG_HIP:" << BOOST_LANG_HIP << std::endl;
#endif
#if BOOST_ARCH_PTX
    std::cout << "BOOST_ARCH_PTX:" << BOOST_ARCH_PTX << std::endl;
#endif
#if BOOST_ARCH_HSA
    std::cout << "BOOST_ARCH_HSA:" << BOOST_ARCH_HSA << std::endl;
#endif
#if BOOST_COMP_NVCC
    std::cout << "BOOST_COMP_NVCC:" << BOOST_COMP_NVCC << std::endl;
#endif
#if BOOST_COMP_HCC
    std::cout << "BOOST_COMP_HCC:" << BOOST_COMP_HCC << std::endl;
#endif
#if BOOST_COMP_CLANG
    std::cout << "BOOST_COMP_CLANG:" << BOOST_COMP_CLANG << std::endl;
#endif
#if BOOST_COMP_GNUC
    std::cout << "BOOST_COMP_GNUC:" << BOOST_COMP_GNUC << std::endl;
#endif
#if BOOST_COMP_MSVC
    std::cout << "BOOST_COMP_MSVC:" << BOOST_COMP_MSVC << std::endl;
#endif
#if BOOST_COMP_CLANG_CUDA
    std::cout << "BOOST_COMP_CLANG_CUDA:" << BOOST_COMP_CLANG_CUDA << std::endl;
#endif
#if BOOST_LIB_STD_GNU
    std::cout << "BOOST_LIB_STD_GNU:" << BOOST_LIB_STD_GNU << std::endl;
#endif
#if BOOST_LIB_STD_CXX
    std::cout << "BOOST_LIB_STD_CXX:" << BOOST_LIB_STD_CXX << std::endl;
#endif
#if BOOST_LIB_STD_DINKUMWARE
    std::cout << "BOOST_LIB_STD_DINKUMWARE:" << BOOST_LIB_STD_DINKUMWARE << std::endl;
#endif
}
