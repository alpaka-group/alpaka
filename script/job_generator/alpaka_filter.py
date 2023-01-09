"""Copyright 2023 Simeon Ehrig

This file is part of alpaka.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Alpaka project specific filter rules.
"""

from typing import List

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

from alpaka_job_coverage.util import (
    row_check_name,
    row_check_version,
    row_check_backend_version,
)


def alpaka_post_filter(row: List) -> bool:
    # disable clang as host compiler for nvcc 11.3 until 11.5
    # https://github.com/alpaka-group/alpaka/issues/1625
    if row_check_name(row, DEVICE_COMPILER, "==", NVCC) and row_check_name(
        row, HOST_COMPILER, "==", CLANG
    ):
        if row_check_version(row, DEVICE_COMPILER, ">=", "11.3") and row_check_version(
            row, DEVICE_COMPILER, "<=", "11.5"
        ):
            return False

    # disable all clang versions older than 14 as CUDA Compiler
    # https://github.com/alpaka-group/alpaka/issues/1857
    if row_check_backend_version(
        row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF
    ) and row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA):
        if row_check_version(row, DEVICE_COMPILER, "<", "14"):
            return False

        # a bug in CMAKE 3.18 avoids the correct usage of the variable CMAKE_CUDA_ARCHITECTURE
        if row_check_version(row, CMAKE, "<", "3.19"):
            return False

    return True
