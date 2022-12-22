"""Copyright 2023 Simeon Ehrig

This file is part of alpaka.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.


Contains alpaka specific function to manipulate the values of a generated job matrix.
"""

from typing import List, Dict, Tuple

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import versions


def set_job_type(job_matrix: List[Dict[str, Tuple[str, str]]]):
    """Decide if a job is a compile only or runtime test job

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): Job matrix
    """
    for hip_version in versions.sw_versions[HIPCC]:
        for job in job_matrix:
            if (
                job[DEVICE_COMPILER][NAME] == HIPCC
                and job[DEVICE_COMPILER][VERSION] == hip_version
                and job[BUILD_TYPE][VERSION] == CMAKE_DEBUG
            ):
                job[TEST_TYPE] = (TEST_TYPE, TEST_RUNTIME)
                break
