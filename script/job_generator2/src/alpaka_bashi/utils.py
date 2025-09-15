"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Utils for the job-generator
"""

from typeguard import typechecked
import bashiValidate


@typechecked
def print_warn(msg: str):
    """Print message in yellow with a [WARNING] prefix.

    Args:
        msg (str): warning text
    """
    print(bashiValidate.utils.cs(f"[WARNING]: {msg}", bashiValidate.utils.Color.YELLOW))
