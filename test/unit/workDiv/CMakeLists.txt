# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
#
# SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
# SPDX-FileContributor: Antonio Di Pilato <tony.dipilato03@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

set(_TARGET_NAME "workDivTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
