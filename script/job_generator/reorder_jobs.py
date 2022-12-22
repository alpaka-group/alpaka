"""Copyright 2023 Simeon Ehrig

This file is part of alpaka.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.


Functions to modify order of the job list.
"""

from typing import List, Dict, Tuple

from typeguard import typechecked

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import search_and_move_job
from versions import sw_versions


@typechecked
def reorder_jobs(job_matrix: List[Dict[str, Tuple[str, str]]]):
    """Vikunja specific function, to move jobs, which matches certain properties to the first waves.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job_matrix.
    """
    pass
