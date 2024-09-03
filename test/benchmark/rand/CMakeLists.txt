#
# Copyright 2022 Jiri Vyskocil
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

set(_TARGET_NAME "randBenchmark")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/benchmark")
target_compile_definitions(${_TARGET_NAME} PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)

if(alpaka_CI)
	# For non-benchmarking CI test runs - It will only run the benchmark once to see if it works at all.
	add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS} --benchmark-samples 1)
  	# Real automated benchmark runs will need to collect more samples (the default 100 is fine). The CI will then 
  	# have to set another variable to indicate if it is only testing, or if it wants to do a full benchmark. 
else()
	# For full benchmark run - will collect 100 samples for good benchmark statistics.
	add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
endif()
