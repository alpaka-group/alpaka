"""Copyright 2023 Simeon Ehrig

This file is part of alpaka.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.


This module contains constants used for the alpaka job generation.
"""

from typing import List

# additional alpaka specific parameters
BUILD_TYPE: str = "build_type"
TEST_TYPE: str = "test_type"

# possible values of BUILD_TYPE
CMAKE_RELEASE: str = "Release"
CMAKE_DEBUG: str = "Debug"
BUILD_TYPES: List[str] = [CMAKE_RELEASE, CMAKE_DEBUG]

# possible values of TEST_TYPE
TEST_RUNTIME: str = "runtime"
TEST_COMPILE_ONLY: str = "compile_only"
