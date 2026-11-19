################################################################################
# Copyright 2025 Your Name
# SPDX-License-Identifier: MPL-2.0
################################################################################

cmake_minimum_required(VERSION 3.22)

set(_TARGET_NAME vectorAddSIMD)

append_recursive_files_add_to_src_group("src" "src" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC "alpaka::alpaka")
