"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the variables of the GitLab CI test job yaml.
"""

from typing import Dict, Any
from typeguard import typechecked
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.globals
from alpaka_bashi.globals import JOB_EXECUTION_TYPE


@typechecked
def set_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set the variables of the GitLab CI test job yaml depending on the combination.

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
    if "variables" not in job_body:
        job_body["variables"] = {}

    job_body["variables"]["JOB_EXECUTION_TYPE"] = str(
        alpaka_bashi.globals.get_version_aliases()[JOB_EXECUTION_TYPE][
            combination[JOB_EXECUTION_TYPE].version
        ],
    )
