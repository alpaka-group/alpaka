#
# Copyright 2025 René Widera
# SPDX-License-Identifier: MPL-2.0
#

### Provide the alpaka target names linked to a target
##
## All targets will recursively be resolved to the actual targets linked to it.
## Output names are: ALPAKA, CUDA, HIP, ONEAPI, HEADERS, HOST
##
## processed_targets_var:
##   Name of a list variable containing targets that were already processed.
##   The updated list is returned through the same variable.
##
## out_var:
##   The detected alpaka target names are appended to this list variable.
function(alpaka_get_targets_recursive target processed_targets_var out_var)
    set(processed_targets "${${processed_targets_var}}")
    set(alpaka_target_desc "${${out_var}}")

    # Stop recursion if this target was already visited. This also breaks
    # dependency cycles such as libA -> libB -> libA.
    if(NOT target IN_LIST processed_targets)
        # Mark the target as processed before following its dependencies.
        list(APPEND processed_targets "${target}")

        get_target_property(libs_linked "${target}" LINK_LIBRARIES)
        get_target_property(libs_linked_interface "${target}" INTERFACE_LINK_LIBRARIES)

        # get_target_property() returns <variable>-NOTFOUND when the property is not set. Do not treat that value
        # as a linked library.
        if(NOT libs_linked)
            set(libs_linked "")
        endif()

        if(NOT libs_linked_interface)
            set(libs_linked_interface "")
        endif()

        set(libs "")
        list(APPEND libs ${libs_linked_interface} ${libs_linked})

        foreach(lib IN LISTS libs)
            # Check all linked targets in case alpaka is linked transitively.
            if(TARGET "${lib}")
                # Use a temporary variable so the recursive call can extend the processed
                # target list and return it to this scope.
                set(sub_targets "")

                # Pass the current processed-target list into the recursive call.
                set(sub_processed_targets "${processed_targets}")
                alpaka_get_targets_recursive(
                        "${lib}"
                        sub_processed_targets
                        sub_targets
                )

                # Propagate all targets visited by the recursive call.
                set(processed_targets "${sub_processed_targets}")
                list(APPEND alpaka_target_desc ${sub_targets})
            endif()

            # Check whether one of the following alpaka targets is linked.
            if(lib MATCHES "alpaka::alpaka|^alpaka$")
                list(APPEND alpaka_target_desc "ALPAKA")
            elseif(lib MATCHES "alpaka_target_cuda|alpaka::cuda")
                list(APPEND alpaka_target_desc "CUDA")
            elseif(lib MATCHES "alpaka_target_hip|alpaka::hip")
                list(APPEND alpaka_target_desc "HIP")
            elseif(lib MATCHES "alpaka_target_oneapi|alpaka::oneapi")
                list(APPEND alpaka_target_desc "ONEAPI")
            elseif(lib MATCHES "alpaka_target_host|alpaka::host")
                list(APPEND alpaka_target_desc "HOST")
            elseif(lib MATCHES "alpaka_target_headers|alpaka::headers")
                list(APPEND alpaka_target_desc "HEADERS")
            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES processed_targets)
    list(REMOVE_DUPLICATES alpaka_target_desc)

    # Return both updated lists to the caller.
    set(${processed_targets_var} "${processed_targets}" PARENT_SCOPE)
    set(${out_var} "${alpaka_target_desc}" PARENT_SCOPE)
endfunction()

### Provide the alpaka target names linked to a target
##
## out_var will contain a list of alpaka target names linked to the target.
function(alpaka_get_targets target out_var)
    set(processed_targets "")
    set(alpaka_target_desc "")

    alpaka_get_targets_recursive("${target}" processed_targets alpaka_target_desc)

    message(DEBUG "alpaka targets for '${target}': ${alpaka_target_desc}; " "processed targets: ${processed_targets}")

    # Return the list via the output variable.
    set(${out_var} "${alpaka_target_desc}" PARENT_SCOPE)
endfunction()

### Copy a source file to the build tree
##
## The copied file will be placed in a subdirectory:
## ${CMAKE_BINARY_DIR}/alpaka_build_files/${api_name}/${PROJECT_NAME}/<relative_to_project_root>/${SRC_FILE_NAME}
## All source file properties set on the original file will be copied to the new file.
## The path including the file name of the copied file is returned via the OUT_VAR variable.
##
## Copying the original file to a unique destination is required to set language properties per file.
## A file can only have one language property but alpaka support using different languages for the same files of a target.
##
## example:
##   CMAKE_BINARY_DIR=build (defined by CMake configure)
##   PROJECT_NAME=vectorAdd (defined by CMake configure)
##
##   SRC_FILE=alpaka/example/src/vector_add.cpp
##   api_name=cuda
##   OUT_VAR=alpaka/build/alpaka_build_files/cuda/vectorAdd/example/src/vector_add.cpp
function(copy_with_structure SRC_FILE api_name OUT_VAR)
    # get the absolut path to derive in the next command the relative path to CMAKE_CURRENT_LIST_DIR
    get_filename_component(SRC_FILE_ABSOLUTE "${SRC_FILE}" REALPATH)
    file(RELATIVE_PATH SRC_FILE_RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${SRC_FILE_ABSOLUTE})

    get_filename_component(SRC_DIR ${SRC_FILE_RELATIVE} DIRECTORY)
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" REL_PATH_TMP "${SRC_DIR}")

    # If the destination directory and the source file have not a common directory e.g. because of symlink usage
    # we replace ../ with ./ to avoid that we try to create the destination file somewhere in the filesystem where
    # we maybe do not have write access.
    string(REPLACE "../" "./" REL_PATH "${REL_PATH_TMP}")

    get_filename_component(SRC_FILE_NAME ${SRC_FILE} NAME)

    # Destination file inside build tree
    set(DEST_FILE ${CMAKE_BINARY_DIR}/alpaka_build_files/${api_name}/${PROJECT_NAME}/${REL_PATH}/${SRC_FILE_NAME})

    # Ensure directory exists
    get_filename_component(DEST_DIR ${DEST_FILE} DIRECTORY)
    file(MAKE_DIRECTORY ${DEST_DIR})

    # Copy file
    configure_file(${SRC_FILE} ${DEST_FILE} COPYONLY)

    # Query all properties actually set on the original file
    get_source_file_property(props ${SRC_FILE} PROPERTY_LIST)

    # Copy properties over the the destination file
    if(props)
        foreach(prop IN LISTS props)
            get_source_file_property(val ${SRC_FILE} ${prop})
            if(val)
                set_source_files_properties(${DEST_FILE} PROPERTIES ${prop} "${val}")
            endif()
        endforeach()
    endif()

    # Export the DEST_FILE path back to caller
    set(${OUT_VAR} ${DEST_FILE} PARENT_SCOPE)
endfunction()

function(alpaka_internal_finalize target)
    # List of alpaka target descriptions found within CMake targets linked to the input target of this function call.
    set(alpaka_target_list "")
    alpaka_get_targets(${target} alpaka_target_list)

    if(NOT alpaka_target_list)
        message(FATAL_ERROR "No alpaka target linked to target '${target}'")
    endif()

    # The index wil be -1 if the target is not in the list
    list(FIND alpaka_target_list "CUDA" index_cuda)
    list(FIND alpaka_target_list "HIP" index_hip)
    list(FIND alpaka_target_list "ONEAPI" index_oneapi)
    list(FIND alpaka_target_list "HEADERS" index_headers)
    list(FIND alpaka_target_list "ALPAKA" index_alpaka)
    list(FIND alpaka_target_list "HOST" index_host)

    # CUDA and HIP can not be used together
    if((NOT index_cuda EQUAL -1) AND (NOT index_hip EQUAL -1))
        message(FATAL_ERROR "Target '${target}' links gainst both CUDA and HIP alpaka targets. Please choose only one.")
    endif()
    # CUDA or HIP cannot be used together with OneAPI
    if(((NOT index_cuda EQUAL -1) OR (NOT index_hip EQUAL -1)) AND (NOT index_oneapi EQUAL -1))
        message(
            FATAL_ERROR
            "Target '${target}' links gainst both CUDA/HIP and OneAPi alpaka targets. Please choose only one."
        )
    endif()

    # You should not call alpaka_finalize if you have not linked any alpaka target
    if(
        index_cuda EQUAL -1
        AND index_hip EQUAL -1
        AND index_oneapi EQUAL -1
        AND index_headers EQUAL -1
        AND index_alpaka EQUAL -1
        AND index_host EQUAL -1
    )
        message(
            FATAL_ERROR
            "alpaka_finalize() was called for target '${target}', but no alpaka targets are linked. Linked interface targets: '${_file_list}'"
        )
    endif()

    ### Get the sources files and patch their LANGUAGE
    ##
    ##  For CUDA and HIP the source files need to be copied to a different location in the build tree because it is not possible to set two languages for a single source file.
    ##  File properties are globally visible, so we cannot just set the LANGUAGE property to CUDA or HIP for the original source file.
    ##  For the target the original file list is copied.
    ##  Files which need to be compiled with CUDA or HIP will be replaced by the copied file, all other files remain unchanged.
    ##  When language separation is enabled, files compiled with language CUDA or HIP are copied to separate build-tree locations; otherwise original files
    ##  are retained.
    get_target_property(_file_list ${target} SOURCES)
    set(_new_file_list ${_file_list})
    foreach(_file ${_file_list})
        if((${_file} MATCHES "\\.cpp$") OR (${_file} MATCHES "\\.cxx$") OR (${_file} MATCHES "\\.cc$"))
            # set the default language to CXX and overwrite it below if necessary
            set_source_files_properties(${_file} PROPERTIES LANGUAGE CXX)
        endif()
        if(NOT index_cuda EQUAL -1)
            if(
                (${_file} MATCHES "\\.cpp$")
                OR (${_file} MATCHES "\\.cxx$")
                OR (${_file} MATCHES "\\.cc$")
                OR (${_file} MATCHES "\\.cu$")
            )
                if(alpaka_LANGUAGE_SEPARATION)
                    copy_with_structure(${_file} "cuda" COPIED_FILE)
                    set_source_files_properties(${COPIED_FILE} PROPERTIES LANGUAGE CUDA)
                    list(REMOVE_ITEM _new_file_list ${_file})
                    list(APPEND _new_file_list ${COPIED_FILE})
                    # Update the list of files for the target
                    set_property(TARGET ${target} PROPERTY SOURCES ${_new_file_list})
                else()
                    # We do not need to copy the files because the user explicitly ask to use the original CXX file.
                    set_source_files_properties(${_file} PROPERTIES LANGUAGE CUDA)
                endif()
            endif()
        endif()
        if(NOT index_hip EQUAL -1)
            if(
                (${_file} MATCHES "\\.cpp$")
                OR (${_file} MATCHES "\\.cxx$")
                OR (${_file} MATCHES "\\.cc$")
                OR (${_file} MATCHES "\\.hip$")
            )
                if(alpaka_LANGUAGE_SEPARATION)
                    copy_with_structure(${_file} "hip" COPIED_FILE)
                    set_source_files_properties(${COPIED_FILE} PROPERTIES LANGUAGE HIP)
                    list(REMOVE_ITEM _new_file_list ${_file})
                    list(APPEND _new_file_list ${COPIED_FILE})
                    # Update the list of files for the target
                    set_property(TARGET ${target} PROPERTY SOURCES ${_new_file_list})
                else()
                    # We do not need to copy the files because the user explicitly ask to use the original CXX file.
                    set_source_files_properties(${_file} PROPERTIES LANGUAGE HIP)
                endif()
            endif()
        endif()
    endforeach()

    ### Set target properties
    ##
    ##  For CUDA and HIP some target properties cannot be propagated by the alpaka targets.
    ##  For any target the corresponding target finalize define is set to disable the alpaka_finalize guard within the alpaka include files.
    ##  The guards avoid that a user is including alpaka headers without calling alpaka_finalize() first.

    if(NOT index_cuda EQUAL -1)
        target_compile_definitions(${target} PRIVATE ALPAKA_CMAKE_TARGET_CUDA_FINALIZE_CALLED)

        # We have to set this here since CUDA_SEPARABLE_COMPILATION is not propagated by the alpaka targets.
        # This option can also not set as file property.
        if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
            set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
            set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
        endif()
    endif()
    if(NOT index_hip EQUAL -1)
        target_compile_definitions(${target} PRIVATE ALPAKA_CMAKE_TARGET_HIP_FINALIZE_CALLED)

        # We have to set this here because CMake currently doesn't provide hip_std_${VERSION} for
        # target_compile_features() and HIP_STANDARD isn't propagated by interface libraries.
        set_target_properties(${target} PROPERTIES HIP_STANDARD ${alpaka_CXX_STANDARD} HIP_STANDARD_REQUIRED ON)
    endif()
    if(NOT index_oneapi EQUAL -1)
        target_compile_definitions(${target} PRIVATE ALPAKA_CMAKE_TARGET_ONEAPI_FINALIZE_CALLED)
    endif()
    if(NOT index_headers EQUAL -1)
        target_compile_definitions(${target} PRIVATE ALPAKA_CMAKE_TARGET_HEADERS_FINALIZE_CALLED)
    endif()
    if(NOT index_host EQUAL -1)
        target_compile_definitions(${target} PRIVATE ALPAKA_CMAKE_TARGET_HOST_FINALIZE_CALLED)
    endif()
    if(NOT index_alpaka EQUAL -1)
        target_compile_definitions(${target} PRIVATE ALPAKA_CMAKE_TARGET_ALPAKA_FINALIZE_CALLED)
    endif()

    # conditionally add coverage and sanitizer support if not compiling with cuda/hip/oneapi
    if(index_cuda EQUAL -1 AND index_hip EQUAL -1 AND index_oneapi EQUAL -1)
        if(alpaka_COVERAGE)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
                message(STATUS "Enabling coverage instrumentation for ${target}")
                target_compile_options(${target} PRIVATE --coverage)
                target_link_options(${target} PRIVATE --coverage)
            else()
                message(
                    WARNING
                    "Coverage instrumentation requested for '${target}', but compiler '${CMAKE_CXX_COMPILER_ID}' is not supported."
                )
            endif()
        endif()
        if(alpaka_ASAN)
            message(STATUS "Linking ASAN to ${target}")
            target_compile_options(${target} PRIVATE -fsanitize=address)
            target_link_options(${target} PRIVATE -fsanitize=address)
            if(alpaka_TSAN OR alpaka_LSAN OR alpaka_UBSAN)
                message(
                    WARNING
                    "Multiple sanitizers are enabled, but only one can be linked at a time. Using ASAN only."
                )
            endif()
        elseif(alpaka_TSAN)
            message(STATUS "Linking TSAN to ${target}")
            target_compile_options(${target} PRIVATE -fsanitize=thread)
            target_link_options(${target} PRIVATE -fsanitize=thread)
            # - to get nicer stack-traces:
            target_link_options(${target} PRIVATE -fno-omit-frame-pointer)
            if(alpaka_LSAN OR alpaka_UBSAN)
                message(
                    WARNING
                    "Multiple sanitizers are enabled, but only one can be linked at a time. Using TSAN only."
                )
            endif()
        elseif(alpaka_LSAN)
            message(STATUS "Linking LSAN to ${target}")
            target_compile_options(${target} PRIVATE -fsanitize=leak)
            target_link_options(${target} PRIVATE -fsanitize=leak)
            if(alpaka_UBSAN)
                message(
                    WARNING
                    "Multiple sanitizers are enabled, but only one can be linked at a time. Using LSAN only."
                )
            endif()
        elseif(alpaka_UBSAN)
            message(STATUS "Linking UBSAN to ${target}")
            target_compile_options(${target} PRIVATE -fsanitize=undefined)
            target_link_options(${target} PRIVATE -fsanitize=undefined)
        endif()
    endif()
endfunction()

### Update target properties and source file properties to enable compilation with the selected alpaka targets.
##
## This function must be called after all alpaka targets have been linked to the target.
## It will set the LANGUAGE property of source files to CUDA or HIP if necessary.
## Depending on the linked targets some source files will be copied to a different location in the build tree to avoid conflicts with the LANGUAGE property.
## All source file properties added to the original source files before the finalize call will be copied to the copied files.
## Attention: Source file properties added to the original source files after the finalize call can not be captured and will not forward to the compiler.
##
## Calling this method twice for the same target will result in an undefined behaviour.
## Linking non alpaka targets after calling this method is allowed.
## If new source files are added to the target after calling this method they will not be handled by alpaka.
## In cases where alpaka is used via cmake fetch content or add_subdirectory the languages depending on alpaka's CMake flags will be loaded.
macro(alpaka_finalize target)
    if(NOT _alpaka_ROOT_DIR)
        include("${alpaka_SOURCE_DIR}/cmake/alpakaPrepareForAddSubdirectoryUsage.cmake")
    endif()
    alpaka_internal_finalize(${target})
endmacro()
