#
# CMake configuration for matrixMulAdd example
#

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(_TARGET_NAME matrixMulAdd)

project(${_TARGET_NAME} LANGUAGES CXX)

# This example requires CUDA support for cuBLAS.
if(NOT alpaka_ACC_GPU_CUDA_ENABLE)
    message(WARNING "Skipping build of 'matrixMulAdd' because alpaka_ACC_GPU_CUDA_ENABLE is not enabled.")
    return()
endif()

# Add cuBLAS library
find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cublas)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

alpaka_add_executable(
    ${_TARGET_NAME}
    src/matrixMulAdd.cpp)

target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka ${CUDA_LIBRARIES})

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
