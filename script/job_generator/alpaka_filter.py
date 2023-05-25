"""Copyright 2023 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Alpaka project specific filter rules.
"""

from typing import List

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


def alpaka_post_filter(row: List) -> bool:
    return True
