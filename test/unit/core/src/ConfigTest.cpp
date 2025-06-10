/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/Config.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

TEST_CASE("printDefines", "[core]")
{
#if ALPAKA_LANG_CUDA
    std::cout << "ALPAKA_LANG_CUDA:" << ALPAKA_LANG_CUDA << std::endl;
#endif
#if ALPAKA_LANG_HIP
    std::cout << "ALPAKA_LANG_HIP:" << ALPAKA_LANG_HIP << std::endl;
#endif
#if ALPAKA_ARCH_PTX
    std::cout << "ALPAKA_ARCH_PTX:" << ALPAKA_ARCH_PTX << std::endl;
#endif
#if ALPAKA_COMP_NVCC
    std::cout << "ALPAKA_COMP_NVCC:" << ALPAKA_COMP_NVCC << std::endl;
#endif
#if ALPAKA_COMP_HIP
    std::cout << "ALPAKA_COMP_HIP:" << ALPAKA_COMP_HIP << std::endl;
#endif
#if ALPAKA_COMP_CLANG
    std::cout << "ALPAKA_COMP_CLANG:" << ALPAKA_COMP_CLANG << std::endl;
#endif
#if ALPAKA_COMP_GNUC
    std::cout << "ALPAKA_COMP_GNUC:" << ALPAKA_COMP_GNUC << std::endl;
#endif
#if ALPAKA_COMP_CLANG_CUDA
    std::cout << "ALPAKA_COMP_CLANG_CUDA:" << ALPAKA_COMP_CLANG_CUDA << std::endl;
#endif
}
