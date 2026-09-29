#
# Copyright 2025 Anton Reinhard
# SPDX-License-Identifier: ISC
#

# ###############################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ###############################################################################
# Project.
set(_TARGET_NAME scan)

project(tutorial LANGUAGES CXX)

# -------------------------------------------------------------------------------
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

set(HAS_CUDA_CUB 0)
# alpaka probes CUDA with check_language(CUDA),
# which only sets CMAKE_CUDA_COMPILER if a valid CUDA compiler/toolchain is provided.
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
        message(STATUS "Optional CUDAToolkit not found, cub scan implementation will be disabled.")
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
            set(HAS_CUDA_CUB 1)
        endif()
    endif()
endif()

set(HAS_HIP_CUB 0)
# alpaka probes HIP with check_language(HIP),
# which only sets CMAKE_HIP_COMPILER if a valid HIP compiler/toolchain is provided.
if(CMAKE_HIP_COMPILER)
    find_package(hip)
    find_package(hipCUB)
    if(NOT hip_FOUND)
        message(STATUS "Optional hip not found, hipcub scan implementation will be disabled.")
    elseif(NOT TARGET hip::hipcub)
        message(STATUS "Found optional hip, but not hipcub; hipcub scan implementation will be disabled.")
    else()
        set(HAS_HIP_CUB 1)
        message(STATUS "Found optional hip")
    endif()
endif()

# -------------------------------------------------------------------------------
# Add executable.
file(GLOB_RECURSE scanSource "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(scanFile ${scanSource})
    get_filename_component(scanFileName ${scanFile} NAME)
    string(REPLACE ".cpp" "" scanName ${scanFileName})
    add_executable(${scanName} ${scanFile})
    target_include_directories(${scanName} PRIVATE src)
    target_link_libraries(${scanName} PUBLIC alpaka::alpaka)
    alpaka_finalize(${scanName})
    set_target_properties(${scanName} PROPERTIES FOLDER tutorial)
    target_compile_features(${scanName} PRIVATE cxx_std_20)

    if(HAS_CUDA_CUB)
        target_link_libraries(${scanName} PRIVATE CUDA::cudart_static)
        target_compile_definitions(${scanName} PRIVATE ALPAKA_HAS_CUB=1)
    endif()
    if(HAS_HIP_CUB)
        target_link_libraries(${scanName} PRIVATE hip::hipcub)
        target_compile_definitions(${scanName} PRIVATE ALPAKA_HAS_HIPCUB=1)
    endif()

    add_test(NAME ${scanName} COMMAND ${scanName})
endforeach()
