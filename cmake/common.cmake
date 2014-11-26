#------------------------------------------------------------------------------
# Gets all recursive files with the given ending in the given directory and recursively below.
# This makes adding files easier because we do not have to update a list each time a file is added but this prevents CMake from detecting if it should be rerun!
#------------------------------------------------------------------------------
FUNCTION(append_recursive_files In_RelativeRootDir In_FileExtension Out_VariableAllFilePathsList)
		#MESSAGE(${In_RelativeRootDir})
		#MESSAGE(${In_FileExtension})
	# Get all recursive files.
	FILE(GLOB_RECURSE allFilePathsList "${CMAKE_CURRENT_LIST_DIR}/${In_RelativeRootDir}*.${In_FileExtension}")
		#MESSAGE( "allFilePathsList: ${allFilePathsList}" )
	# Set the return value.
	SET(${Out_VariableAllFilePathsList} ${${Out_VariableAllFilePathsList}} "${allFilePathsList}" PARENT_SCOPE)
ENDFUNCTION()

#------------------------------------------------------------------------------
# Gets all recursive relative subdirectories.
#------------------------------------------------------------------------------
FUNCTION(append_recursive_relative_subdirectories In_RelativeRootDir Out_VariableAllSubDirPathsList)
		#MESSAGE("In_RelativeRootDir: ${In_RelativeRootDir}")
		#MESSAGE("CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
	SET(rootDir "${CMAKE_CURRENT_LIST_DIR}/${In_RelativeRootDir}")
	# Get all the recursive files with their relative paths. 
	FILE(GLOB_RECURSE recursiveFilePathsList RELATIVE ${rootDir} "${rootDir}*")
		#MESSAGE("recursiveFilePathsList: ${recursiveFilePathsList}")
		
	# Get the paths to all the recursive files.
	SET(allSubDirPathsList "")	# Create empty list for the case of no subdirectries being present.
	FOREACH(filePath ${recursiveFilePathsList})
		GET_FILENAME_COMPONENT(dirPath ${filePath} PATH)
		SET(allSubDirPathsList ${allSubDirPathsList} ${dirPath})
	ENDFOREACH()
		#MESSAGE("allSubDirPathsList: ${allSubDirPathsList}")
	
	# If the list is not empty.
	IF(allSubDirPathsList)
		# Remove duplicates from the list.
		LIST(REMOVE_DUPLICATES allSubDirPathsList)
			#MESSAGE("allSubDirPathsList: ${allSubDirPathsList}")
			
		# Append "/" to all the paths.
		SET(allSubDirsList "")
		FOREACH(filePath ${allSubDirPathsList})
			LIST(APPEND allSubDirsList "${filePath}/")
		ENDFOREACH()
			#MESSAGE("allSubDirsList: ${allSubDirsList}")
		
		# Set the return value.
			#MESSAGE("${Out_VariableAllSubDirPathsList}")
		SET(${Out_VariableAllSubDirPathsList} ${${Out_VariableAllSubDirPathsList}} "${allSubDirsList}" PARENT_SCOPE)
	ENDIF()
ENDFUNCTION()

#------------------------------------------------------------------------------
# Groups the files in the same way the directories are structured.
# This makes adding files easier because we do not have to update a list each time a file is added but this prevents CMake from detecting if it should be rerun!
#------------------------------------------------------------------------------
FUNCTION(add_recursive_files_to_source_group In_RelativeRootDir In_FileExtension)
		#MESSAGE(${In_RelativeRootDir})
		#MESSAGE(${In_FileExtension})
	# Get all recursive subdirectories.
	append_recursive_relative_subdirectories("${In_RelativeRootDir}" recursiveRelativeSubDirList)
		#MESSAGE("recursiveRelativeSubDirList: ${recursiveRelativeSubDirList}")

	# For the folder itself and each sub-folder...
	FOREACH(currentRelativeSubDir "" ${recursiveRelativeSubDirList})
		# Get all the files in this sub-folder.
		SET(wildcardFilePath "${CMAKE_CURRENT_LIST_DIR}/${In_RelativeRootDir}${currentRelativeSubDir}*.${In_FileExtension}")
			#MESSAGE("wildcardFilePath: ${wildcardFilePath}")
		FILE(GLOB filesInSubDirList ${wildcardFilePath})
			#MESSAGE("filesInSubDirList: ${filesInSubDirList}")
			
		# Group the include files into a project sub-folder analogously to the filesystem hierarchy.
		SET(groupExpression "${In_RelativeRootDir}${currentRelativeSubDir}")
			#MESSAGE("groupExpression: ${groupExpression}")
		# Remove the parent directory steps from the path.
		#STRING(REPLACE "../" "" groupExpression "${groupExpression}")
			#MESSAGE("groupExpression: ${groupExpression}")
		# Replace the directory separators in the path to build valid grouping expressions.
		STRING(REPLACE "/" "\\" groupExpression "${groupExpression}")
			#MESSAGE("groupExpression: ${groupExpression}")
		SOURCE_GROUP("${groupExpression}" FILES ${filesInSubDirList})
	ENDFOREACH()
ENDFUNCTION()

#------------------------------------------------------------------------------
# Gets all files with the given ending in the given directory.
# Groups the files in the same way the directories are structured.
# This makes adding files easier because we do not have to update a list each time a file is added but this prevents CMake from detecting if it should be rerun!
#------------------------------------------------------------------------------
FUNCTION(append_recursive_files_add_to_source_group In_RelativeRootDir In_FileExtension Out_VariableAllFilePathsList)
		#MESSAGE(${In_RelativeRootDir})
		#MESSAGE(${In_FileExtension})
		#MESSAGE("CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
	SET(allFilePathsListToAppend) 	# We have to use a local variable and give it to the parent because append_recursive_files only gives it to our scope with PARENT_SCOPE.
	append_recursive_files(${In_RelativeRootDir} ${In_FileExtension} allFilePathsListToAppend)
		#MESSAGE( "allFilePathsListToAppend: ${allFilePathsListToAppend}" )
	add_recursive_files_to_source_group(${In_RelativeRootDir} ${In_FileExtension})
	SET(${Out_VariableAllFilePathsList} ${${Out_VariableAllFilePathsList}} ${allFilePathsListToAppend} PARENT_SCOPE)
ENDFUNCTION()

#------------------------------------------------------------------------------
# void list_add_prefix(string prefix, list<string>* list_of_items);
# - returns list_of_items with prefix prepended to all items
# - original list is modified
#------------------------------------------------------------------------------
FUNCTION(list_add_prefix prefix list_of_items)
    SET(local_list "")
    FOREACH(item IN LISTS "${list_of_items}")
        IF(NOT ${item} STREQUAL "")
            LIST(APPEND local_list "${prefix}${item}")
        ENDIF()
    ENDFOREACH()
    SET(${list_of_items} "${local_list}" PARENT_SCOPE)
ENDFUNCTION()

#------------------------------------------------------------------------------
# void list_add_prefix(string prefix, string item_to_prefix, list<string>* list_of_items);
# - returns list_of_items with prefix prepended to all items
# - original list is modified
#------------------------------------------------------------------------------
FUNCTION(list_add_prefix_to prefix item_to_prefix list_of_items)
    SET(local_list "")
    FOREACH(item IN LISTS "${list_of_items}")
        IF(${item} STREQUAL ${item_to_prefix})
            LIST(APPEND local_list "${prefix}${item}")
		ELSE()
            LIST(APPEND local_list ${item})
        ENDIF()
    ENDFOREACH()
    SET(${list_of_items} "${local_list}" PARENT_SCOPE)
ENDFUNCTION()