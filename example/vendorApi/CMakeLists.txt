#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME vendorApi)

project(${_TARGET_NAME} LANGUAGES CXX)

#-------------------------------------------------------------------------------
# Find alpaka.

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()
# alpaka probes CUDA with check_language(CUDA),
# which only sets CMAKE_CUDA_COMPILER if a valid CUDA compiler/toolchain is provided.
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
        message(STATUS "Optional CUDAToolkit not found, alpaka vendor api will not use thrust")
    else()
        message(STATUS "Found optional CUDAToolkit, version " ${CUDAToolkit_VERSION})
        if(
            CMAKE_CUDA_COMPILER_ID STREQUAL "Clang"
            AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 23.0
            AND ${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "13.1"
            AND ${CUDAToolkit_VERSION} VERSION_LESS "13.3"
        )
            # Clang>=23 can not be used with CCCL due to compile issues
            # https://github.com/NVIDIA/cccl/issues/7896
            # This issue should be fixed with CCCL shipped by CUDA 13.3+
            message(WARNING "CUDAToolkit will not be used due to incompatibilities with the clang CUDA compiler")
        else()
            set(_alpaka_HAS_CUDA_TOOLKIT 1)
        endif()
    endif()
endif()

#-------------------------------------------------------------------------------
# Add executable.

add_executable(${_TARGET_NAME} src/main.cpp)
target_include_directories(${_TARGET_NAME} PRIVATE "src")
target_link_libraries(${_TARGET_NAME} PUBLIC alpaka::alpaka)
alpaka_finalize(${_TARGET_NAME})

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

if(_alpaka_HAS_CUDA_TOOLKIT)
    target_link_libraries(${_TARGET_NAME} PRIVATE CUDA::cudart_static)
endif()

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
