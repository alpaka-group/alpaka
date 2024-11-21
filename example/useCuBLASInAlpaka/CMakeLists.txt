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
find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cublas)

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

alpaka_add_executable(
    ${_TARGET_NAME}
    src/useCuBLASInAlpaka.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka ${CUDA_LIBRARIES})

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)
#set_target_properties(${_TARGET_NAME}  PROPERTIES CUDA_STANDARD 17)
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
