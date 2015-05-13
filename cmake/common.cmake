#
# Copyright 2014-2015 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

#------------------------------------------------------------------------------
# Gets all recursive files with the given ending in the given directory and recursively below.
# This makes adding files easier because we do not have to update a list each time a file is added but this prevents CMake from detecting if it should be rerun!
#------------------------------------------------------------------------------
FUNCTION(append_recursive_files In_RootDir In_FileExtension Out_VariableAllFilePathsList)
        #MESSAGE("In_RootDir: ${In_RootDir}")
        #MESSAGE("In_FileExtension: ${In_FileExtension}")
        #MESSAGE("Out_VariableAllFilePathsList: ${Out_VariableAllFilePathsList}")
    # Get all recursive files.
    FILE(
        GLOB_RECURSE
        allFilePathsList
        "${In_RootDir}*.${In_FileExtension}")
        #MESSAGE( "allFilePathsList: ${allFilePathsList}" )
    # Set the return value.
    SET(
        "${Out_VariableAllFilePathsList}"
        "${${Out_VariableAllFilePathsList}}" "${allFilePathsList}"
        PARENT_SCOPE)
ENDFUNCTION()

#------------------------------------------------------------------------------
# Gets all recursive relative subdirectories.
#------------------------------------------------------------------------------
FUNCTION(append_recursive_relative_subdirs In_RootDir Out_VariableAllSubDirPathsList)
        #MESSAGE("In_RootDir: ${In_RootDir}")
    # Get all the recursive files with their relative paths.
    FILE(
        GLOB_RECURSE
        "recursiveFilePathsList"
        RELATIVE "${In_RootDir}" "${In_RootDir}*")
        #MESSAGE("recursiveFilePathsList: ${recursiveFilePathsList}")

    # Get the paths to all the recursive files.
    SET(
        "allSubDirPathsList")    # Create empty list for the case of no subdirectories being present.
    FOREACH(
        "filePath"
        IN
        LISTS "recursiveFilePathsList")
        GET_FILENAME_COMPONENT(
            "dirPath"
            "${filePath}"
            PATH)
        LIST(
            APPEND
            "allSubDirPathsList"
            "${dirPath}")
    ENDFOREACH()
        #MESSAGE("allSubDirPathsList: ${allSubDirPathsList}")

    # If the list is not empty.
    LIST(
        LENGTH
        "allSubDirPathsList"
        "allSubDirPathsListLength")
    IF("${allSubDirPathsListLength}")
        # Remove duplicates from the list.
        LIST(
            REMOVE_DUPLICATES
            "allSubDirPathsList")
            #MESSAGE("allSubDirPathsList: ${allSubDirPathsList}")

        # Append "/" to all the paths.
        SET(
            "allSubDirsList")
        FOREACH(
            "filePath"
            IN
            LISTS "allSubDirPathsList")
            LIST(
                APPEND
                "allSubDirsList"
                "${filePath}/")
        ENDFOREACH()
            #MESSAGE("allSubDirsList: ${allSubDirsList}")

        # Set the return value.
            #MESSAGE("${Out_VariableAllSubDirPathsList}")
        SET("${Out_VariableAllSubDirPathsList}" "${${Out_VariableAllSubDirPathsList}}" "${allSubDirsList}" PARENT_SCOPE)
    ENDIF()
ENDFUNCTION()

#------------------------------------------------------------------------------
# Groups the files in the same way the directories are structured.
#------------------------------------------------------------------------------
FUNCTION(add_recursive_files_to_src_group In_RootDir In_SrcGroupIgnorePrefix In_FileExtension)
        #MESSAGE("In_RootDir: ${In_RootDir}")
        #MESSAGE("In_SrcGroupIgnorePrefix: ${In_SrcGroupIgnorePrefix}")
        #MESSAGE("In_FileExtension: ${In_FileExtension}")
    # Get all recursive subdirectories.
    append_recursive_relative_subdirs(
        "${In_RootDir}"
        "recursiveRelativeSubDirList")
        #MESSAGE("recursiveRelativeSubDirList: ${recursiveRelativeSubDirList}")

    # For the folder itself and each sub-folder...
    FOREACH(
        "currentRelativeSubDir"
        IN
        LISTS "recursiveRelativeSubDirList")
        # Appended the current subdirectory.
        SET(
            "currentSubDir"
            "${In_RootDir}${currentRelativeSubDir}")
            #MESSAGE("currentSubDir: ${currentSubDir}")
        # Get all the files in this sub-folder.
        SET(
            "wildcardFilePath"
            "${currentSubDir}*.${In_FileExtension}")
            #MESSAGE("wildcardFilePath: ${wildcardFilePath}")
        FILE(
            GLOB
            "filesInSubDirList"
            "${wildcardFilePath}")
            #MESSAGE("filesInSubDirList: ${filesInSubDirList}")


        LIST(
            LENGTH
            "filesInSubDirList"
            "filesInSubDirListLength")
        IF("${filesInSubDirListLength}")
            # Group the include files into a project sub-folder analogously to the filesystem hierarchy.
            SET(
                "groupExpression"
                "${currentSubDir}")
                #MESSAGE("groupExpression: ${groupExpression}")
            # Remove the parent directory steps from the path.
            # NOTE: This is not correct because it does not only replace at the beginning of the string.
            #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
            STRING(
                REPLACE "${In_SrcGroupIgnorePrefix}" ""
                "groupExpression"
                "${groupExpression}")
                #MESSAGE("groupExpression: ${groupExpression}")
            # Replace the directory separators in the path to build valid grouping expressions.
            STRING(
                REPLACE "/" "\\"
                "groupExpression"
                "${groupExpression}")
                #MESSAGE("groupExpression: ${groupExpression}")
            SOURCE_GROUP(
                "${groupExpression}"
                FILES ${filesInSubDirList})
        ENDIF()
    ENDFOREACH()
ENDFUNCTION()

#------------------------------------------------------------------------------
# Gets all files with the given ending in the given directory.
# Groups the files in the same way the directories are structured.
# This makes adding files easier because we do not have to update a list each time a file is added but this prevents CMake from detecting if it should be rerun!
#------------------------------------------------------------------------------
FUNCTION(append_recursive_files_add_to_src_group In_RootDir In_SrcGroupIgnorePrefix In_FileExtension Out_VariableAllFilePathsList)
        #MESSAGE("In_RootDir: ${In_RootDir}")
        #MESSAGE("In_SrcGroupIgnorePrefix: ${In_SrcGroupIgnorePrefix}")
        #MESSAGE("In_FileExtension: ${In_FileExtension}")
        #MESSAGE("Out_VariableAllFilePathsList: ${Out_VariableAllFilePathsList}")
    # We have to use a local variable and give it to the parent because append_recursive_files only gives it to our scope with PARENT_SCOPE.
    SET(
        "allFilePathsList"
        "${${Out_VariableAllFilePathsList}}")
    append_recursive_files(
        "${In_RootDir}"
        "${In_FileExtension}"
        "allFilePathsList")
        #MESSAGE( "allFilePathsList: ${allFilePathsList}" )
    SET(
        "${Out_VariableAllFilePathsList}"
        "${${Out_VariableAllFilePathsList}}" "${allFilePathsList}"
        PARENT_SCOPE)

    add_recursive_files_to_src_group(
        "${In_RootDir}"
        "${In_SrcGroupIgnorePrefix}"
        "${In_FileExtension}")
ENDFUNCTION()

#------------------------------------------------------------------------------
# void list_add_prefix(string prefix, list<string>* list_of_items);
# - returns The list_of_items with prefix prepended to all items.
# - original list is modified
#------------------------------------------------------------------------------
FUNCTION(list_add_prefix prefix list_of_items)
    SET("local_list")

    FOREACH(
        "item"
        IN
        LISTS "${list_of_items}")
        IF(POLICY CMP0054)
            CMAKE_POLICY(SET CMP0054 NEW)   # Only interpret if() arguments as variables or keywords when unquoted.
        ENDIF()
        IF(NOT "${item}" STREQUAL "")
            LIST(
                APPEND
                "local_list"
                "${prefix}${item}")
        ENDIF()
    ENDFOREACH()

    SET(
        "${list_of_items}"
        "${local_list}"
        PARENT_SCOPE)
ENDFUNCTION()

#------------------------------------------------------------------------------
# void list_add_prefix(string prefix, string item_to_prefix, list<string>* list_of_items);
# - returns list_of_items with prefix prepended to all items matching item_to_prefix.
# - original list is modified
#------------------------------------------------------------------------------
FUNCTION(list_add_prefix_to prefix item_to_prefix list_of_items)
    SET("local_list")

    FOREACH(
        "item"
        IN LISTS "${list_of_items}")
        IF(POLICY CMP0054)
            CMAKE_POLICY(SET CMP0054 NEW)   # Only interpret if() arguments as variables or keywords when unquoted.
        ENDIF()
        IF("${item}" STREQUAL "${item_to_prefix}")
            LIST(
                APPEND
                "local_list"
                "${prefix}${item}")
        ELSE()
            LIST(
                APPEND
                "local_list"
                "${item}")
        ENDIF()
    ENDFOREACH()

    SET(
        "${list_of_items}"
        "${local_list}"
        PARENT_SCOPE)
ENDFUNCTION()
