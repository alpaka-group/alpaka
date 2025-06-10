#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME useCuBLASInAlpaka)

project(${_TARGET_NAME} LANGUAGES CXX)

# Check if the cmake variables to see if the Acc option is Cuda Only.
if(NOT alpaka_ACC_GPU_CUDA_ONLY_MODE)
    # Print a warning and skip target creation
    message(WARNING "Skipping build of 'useCuBLASInAlpaka' because alpaka_ACC_GPU_CUDA_ONLY_MODE is not enabled.")
    return()
endif()

# Add cuBLAS library
find_package(CUDAToolkit REQUIRED)
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

# Debug: Print CUDA toolkit information
message(STATUS "CUDAToolkit found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit include directories: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit library directories: ${CUDAToolkit_LIBRARY_DIR}")

# Verify cuBLAS header exists
if(CUDAToolkit_INCLUDE_DIRS)
    find_file(CUBLAS_HEADER_FILE
        NAMES cublas_v2.h
        PATHS ${CUDAToolkit_INCLUDE_DIRS}
        NO_DEFAULT_PATH
    )
    if(NOT CUBLAS_HEADER_FILE)
        message(FATAL_ERROR "cublas_v2.h not found in CUDA toolkit include directories: ${CUDAToolkit_INCLUDE_DIRS}")
    else()
        message(STATUS "Found cublas_v2.h: ${CUBLAS_HEADER_FILE}")
    endif()
else()
    message(WARNING "CUDAToolkit_INCLUDE_DIRS is empty - this may cause compilation issues")
endif()
#-------------------------------------------------------------------------------
# Add executable.

alpaka_add_executable(
    ${_TARGET_NAME}
    src/useCuBLASInAlpaka.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka CUDA::cublas)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)
#set_target_properties(${_TARGET_NAME}  PROPERTIES CUDA_STANDARD 17)
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
