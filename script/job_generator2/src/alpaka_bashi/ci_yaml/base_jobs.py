"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Get GitLab CI basic job yaml's. The job skeleton will be extended and modified for specific job
configurations.
"""

from typing import Dict


def get_dummy_job() -> Dict:
    """Return GitLab CI job, which simply prints a message. Can be used, if no job is generated for
    a CI pipeline."""
    return {
        "dummy-job": {
            "image": "alpine:latest",
            "interruptible": True,
            "script": ['echo "This is a dummy job so that the CI does not fail."'],
        }
    }
