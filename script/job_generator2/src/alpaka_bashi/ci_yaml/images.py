"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the image of the GitLab CI test job yaml.
"""

from typing import Dict, Any
from typeguard import typechecked
import bashi


@typechecked
def set_image(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set the image of the GitLab CI test job yaml depending on the combination.

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
    job_body["image"] = "alpine:latest"
