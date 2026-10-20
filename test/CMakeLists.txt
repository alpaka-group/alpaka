#
# Copyright 2025 René Widera, Simeon Ehrig
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################
cmake_minimum_required(VERSION 3.25)

################################################################################
# Test target.
################################################################################

set(_TEST_TARGET_NAME alpakaTest)
add_library(${_TEST_TARGET_NAME} INTERFACE)
target_include_directories(${_TEST_TARGET_NAME} INTERFACE include)
add_library(alpaka::test ALIAS ${_TEST_TARGET_NAME})

################################################################################
# Add subdirectories.
################################################################################

alpaka_install_catch2()

#------------------------------------------------------------------------------
# Search for sub folders in the given folder.
# Creates a Catch2 test executable for each .cpp file in each sub folder.
# If a include folder is located in a sub folder, add it as include path to the
# target.
# Args:
#  - target_prefix: Prefix for the target name. The name of the executable test
#    file is formed with target_prefix followed by the source file name without
#    extension.
#  - folder: Takes all .cpp files in the specified folder and creates an test
#    executable for each.
function(alpaka_add_test_target target_prefix folder)
    message(DEBUG "test target prefix: ${target_prefix}")

    file(GLOB test_folders ${folder})
    foreach(test_folder ${test_folders})
        if(IS_DIRECTORY ${test_folder})
            message(DEBUG "test folder: ${test_folder}")

            file(GLOB test_sources "${test_folder}/**")

            # iterate over all sub folders
            foreach(source_file ${test_sources})
                # the include folder is a special case, therefore ignore it
                if(source_file STREQUAL "${test_folder}/include")
                    continue()
                endif()

                get_filename_component(source_file_extension ${source_file} EXT)
                get_filename_component(source_file_name ${source_file} NAME_WE)
                # get absolute path of the folder of the source file
                get_filename_component(test_folder_name ${source_file} DIRECTORY)
                # get name of the folder
                get_filename_component(test_folder_name ${test_folder_name} NAME)

                if("${source_file_extension}" STREQUAL ".cpp")
                    set(_test_target_name "${target_prefix}${test_folder_name}_${source_file_name}")
                    message(DEBUG "  add test target ${_test_target_name}: ${source_file}")

                    add_executable(${_test_target_name} ${source_file})
                    get_filename_component(src_dir ${source_file} DIRECTORY)
                    target_include_directories(${_test_target_name} PRIVATE ${src_dir})
                    if(EXISTS "${test_folder}/include")
                        message(STATUS "  add include folder: ${test_folder}/include")
                        target_include_directories(${_test_target_name} PRIVATE ${test_folder}/include)
                    endif()

                    target_link_libraries(
                        ${_test_target_name}
                        PRIVATE Catch2::Catch2WithMain alpaka::alpaka alpaka::test
                    )
                    alpaka_finalize(${_test_target_name})
                    catch_discover_tests(${_test_target_name} EXTRA_ARGS "-w" "NoAssertions")
                endif()
            endforeach()
        endif()
    endforeach()
endfunction()

alpaka_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit/**)

# Expectations come from configuration, independently of compiler feature macros.
target_compile_definitions(
    unit_test_core_openmp
    PRIVATE
        ALPAKA_TEST_EXPECT_OMP=$<BOOL:${alpaka_DEP_OMP}>
        ALPAKA_TEST_EXPECT_CPU_OMP=$<AND:$<BOOL:${alpaka_DEP_OMP}>,$<BOOL:${alpaka_EXEC_CpuOmpBlocks}>,$<BOOL:${alpaka_HOST_Cpu}>>
)
