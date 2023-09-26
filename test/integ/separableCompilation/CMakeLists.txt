# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Antonio Di Pilato <tony.dipilato03@gmail.com>
# SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
# SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
# SPDX-FileContributor: René Widera <r.widera@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

if((NOT alpaka_ACC_GPU_CUDA_ENABLE AND NOT alpaka_ACC_GPU_HIP_ENABLE AND NOT alpaka_ACC_SYCL_ENABLE) OR
   (alpaka_ACC_GPU_CUDA_ENABLE AND (alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)) OR # OFF by default
   (alpaka_ACC_GPU_HIP_ENABLE AND (alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)) OR  # OFF by default
   (alpaka_ACC_SYCL_ENABLE AND NOT (alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)))  # ON by default

    set(_TARGET_NAME "separableCompilationTest")

    append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)
    append_recursive_files_add_to_src_group("include/" "include/" "hpp" _FILES_HEADER)

    alpaka_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE}
        ${_FILES_HEADER})
    target_include_directories(
        ${_TARGET_NAME}
        PRIVATE "include")
    target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE common)

    set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})

endif()
