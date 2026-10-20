#
# Copyright 2025 René Widera, Simeon Ehrig
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################
cmake_minimum_required(VERSION 3.25)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory(example)
add_subdirectory(cheatsheet)
add_subdirectory(dataStorage)

################################################################################
# Build fetchContent example in isolation
################################################################################

# Create a separate build directory for the fetchContent example
# This keeps the example's build environment clean and isolated from the main project
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/docs/snippets/fetchContent/)

# Configure the fetchContent example as a separate CMake project
# This simulates how a user would actually use alpaka via FetchContent
# We need to pass through the C++ compiler since the environment isnt shared with the process
execute_process(
    COMMAND
        ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_SOURCE_DIR}/fetchContent -DCMAKE_BUILD_TYPE=Release -G ${CMAKE_GENERATOR}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs/snippets/fetchContent
    RESULT_VARIABLE SUBPROJECT_CONFIGURE_RESULT
    OUTPUT_VARIABLE SUBPROJECT_CONFIGURE_OUTPUT
    ERROR_VARIABLE SUBPROJECT_CONFIGURE_ERROR
)

if(NOT SUBPROJECT_CONFIGURE_RESULT EQUAL 0)
    message(STATUS "Subproject result:\n${SUBPROJECT_CONFIGURE_RESULT}")
    message(STATUS "Subproject configuration failed:\n${SUBPROJECT_CONFIGURE_OUTPUT}")
    message(FATAL_ERROR "Subproject configuration error:\n${SUBPROJECT_CONFIGURE_ERROR}")
endif()

# Create a custom target to build the fetchContent example
# This will be built as part of the main project's ALL target
add_custom_target(
    fetchContent_build
    ALL
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs/snippets/fetchContent
    COMMENT "Building fetchContent snippet..."
    VERBATIM
)

# Add a test with a hardcoded path that runs the fetchContent example in a clean environment
# This verifies that the example pulled into documentation works correctly and can detect devices
add_test(NAME fetchContent COMMAND ${CMAKE_BINARY_DIR}/docs/snippets/fetchContent/fetchContentExample)
