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

# Ensure CUDAToolkit is found
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDAToolkit not found. Please ensure the CUDA toolkit is installed and accessible.")
endif()# Search for cublas_v2.h in /usr/local using the find command

# Search for cublas.h in /usr/local using the find command
execute_process(
    COMMAND find /usr/local -name "cublas.h" -type f
    OUTPUT_VARIABLE CUBLAS_H_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Check if the file was found and print the result
if(CUBLAS_H_PATHS)
    message(STATUS "Found OLD FILE cublas.h in the following locations:")
    message(STATUS "${CUBLAS_H_PATHS}")
else()
    message(STATUS "cublas.h not found in /usr/local.")
endif()

# Search for cublas_v2.h in /usr/local using the find command
execute_process(
    COMMAND find /usr/local -name "cublas_v2.h" -type f
    OUTPUT_VARIABLE CUBLAS_HEADER_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Check if the file was found and print the result
if(CUBLAS_HEADER_PATHS)
    message(STATUS "Found cublas_v2.h in the following locations:")
    message(STATUS "${CUBLAS_HEADER_PATHS}")

    # Split the results into a list
    string(REPLACE "\n" ";" CUBLAS_HEADER_PATH_LIST ${CUBLAS_HEADER_PATHS})

    # Extract CUDA version
    string(REPLACE "." ";" CUDA_VERSION_LIST ${CUDAToolkit_VERSION})
    list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
    list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)

    # Initialize variables for the selected path
    set(SELECTED_CUBLAS_PATH "")
    set(FALLBACK_CUBLAS_PATH "")

    # Iterate over the found paths
    foreach(PATH ${CUBLAS_HEADER_PATH_LIST})
        if(PATH MATCHES "cuda-${CUDA_VERSION_MAJOR}\\.${CUDA_VERSION_MINOR}")
            # If the path matches the major.minor version, select it
            set(SELECTED_CUBLAS_PATH ${PATH})
            break()
        elseif(PATH MATCHES "cuda")
            # Save a fallback path if no version-specific match is found
            set(FALLBACK_CUBLAS_PATH ${PATH})
        endif()
    endforeach()

    # Use the selected path or fallback if no version-specific match is found
    if(SELECTED_CUBLAS_PATH)
        get_filename_component(CUBLAS_INCLUDE_DIR ${SELECTED_CUBLAS_PATH} DIRECTORY)
        message(STATUS "Using version-specific cublas_v2.h from: ${CUBLAS_INCLUDE_DIR}")
    elseif(FALLBACK_CUBLAS_PATH)
        get_filename_component(CUBLAS_INCLUDE_DIR ${FALLBACK_CUBLAS_PATH} DIRECTORY)
        message(STATUS "Using fallback cublas_v2.h from: ${CUBLAS_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "No suitable cublas_v2.h found matching the CUDA version ${CUDAToolkit_VERSION}")
    endif()
else()
    message(FATAL_ERROR "cublas_v2.h not found in /usr/local. Please ensure the CUDA toolkit is installed correctly.")
endif()

# Add executable
alpaka_add_executable(
    ${_TARGET_NAME}
    src/useCuBLASInAlpaka.cpp)

# Add the include directory to the target (after the target is created)
if(CUBLAS_INCLUDE_DIR)
    target_include_directories(${_TARGET_NAME} PRIVATE ${CUBLAS_INCLUDE_DIR})
endif()

# Link cuBLAS library
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka CUDA::cublas)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
