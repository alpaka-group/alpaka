#
# Copyright 2024 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

set(_TARGET_NAME "memTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")
target_compile_definitions(${_TARGET_NAME} PRIVATE "-DTEST_UNIT_MEM")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})


add_subdirectory("buf/")
add_subdirectory("copy/")
add_subdirectory("fence/")
add_subdirectory("p2p/")
add_subdirectory("view/")
