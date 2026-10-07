#
# Copyright 2025 Tapish Narwal
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

project(fetchContent LANGUAGES CXX VERSION 1.0)

# Use CMake's FetchContent to download and integrate alpaka directly from GitHub
include(FetchContent)

# Declare where to fetch alpaka from
# This will download the library at configure time
FetchContent_Declare(alpaka GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git GIT_TAG dev_v3.x)

# Make alpaka available for use in this project
# This downloads, configures, and makes the library targets available
FetchContent_MakeAvailable(alpaka)

# Define the device specification for your application
# Format: "api:deviceKind" where:
#   - api: The parallel programming interface (host, cuda, hip, oneApi)
#   - deviceKind: The type of device (cpu, nvidiaGpu, amdGpu, intelGpu)
# Default is set to "host:cpu" for CPU-only execution
set(myApp_DEVICE_SPEC "host:cpu" CACHE STRING "Selected alpaka Device Specification")

# Define all available device specifications for the cache variable
# This creates a dropdown menu in CMake GUIs for easy selection
set_property(
    CACHE myApp_DEVICE_SPEC
    PROPERTY STRINGS "host:cpu;cuda:nvidiaGpu;hip:amdGpu;oneApi:cpu;oneApi:intelGpu;oneApi:nvidiaGpu;oneApi:amdGpu"
)

# Parse the device specification string into separate API and device kind components
# Split the string at the colon delimiter
string(REPLACE ":" ";" _myAPP_DEVICE_SPEC_LIST "${myApp_DEVICE_SPEC}")

# Extract the API (first element) and device kind (second element) components from the list
list(GET _myAPP_DEVICE_SPEC_LIST 0 _myApp_API)
list(GET _myAPP_DEVICE_SPEC_LIST 1 _myApp_DEVICE_KIND)

# Create the executable target from the source file
add_executable(fetchContentExample main.cpp)

# Link the alpaka library to the executable
target_link_libraries(fetchContentExample PRIVATE alpaka::alpaka)

# Pass the device specification to the C++ code as compile-time definitions
# These will be available as MYAPP_API and MYAPP_DEVICE_KIND macros in the code
target_compile_definitions(fetchContentExample PRIVATE MYAPP_API=${_myApp_API} MYAPP_DEVICE_KIND=${_myApp_DEVICE_KIND})

# Finalize the alpaka configuration for this target
# This sets up backend-specific compiler flags and dependencies
alpaka_finalize(fetchContentExample)
