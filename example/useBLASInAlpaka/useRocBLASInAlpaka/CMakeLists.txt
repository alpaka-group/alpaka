#
# Copyright 2024 Mehmet Yusufoglu, Simeon Ehrig, René Widera
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME useRocBLASInAlpaka)

project(${_TARGET_NAME} LANGUAGES CXX)

if(NOT alpaka_ACC_GPU_HIP_ONLY_MODE)
    # Print a warning and skip target creation
    message(WARNING "Skipping build of 'useRocBLASInAlpaka' because alpaka_ACC_GPU_HIP_ONLY_MODE is not enabled.")
    return()
endif()


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

#-------------------------------------------------------------------------------
# Add executable.
#-------------------------------------------------------------------------------
# Locate rocBLAS.
find_package(rocblas REQUIRED CONFIG )
# Use the line below if the rocblas configuration files (e.g., rocblasConfig.cmake) are not in the default search paths.
# find_package(rocblas REQUIRED CONFIG HINTS /opt/rocm /opt/rocm-<version>)

alpaka_add_executable(
    ${_TARGET_NAME}
    src/useRocBLASInAlpaka.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka rocblas)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
