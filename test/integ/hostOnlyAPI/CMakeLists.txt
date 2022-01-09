#
# Copyright 2022 Andrea Bocci
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

set(_TARGET_NAME "hostOnlyAPITest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

add_compile_definitions(ALPAKA_HOST_API)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

if(ALPAKA_ACC_GPU_CUDA_ENABLE OR ALPAKA_ACC_GPU_HIP_ENABLE)
    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})
endif()
