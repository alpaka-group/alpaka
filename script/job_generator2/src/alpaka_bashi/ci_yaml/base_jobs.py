"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Get GitLab CI basic job yaml's. The job skeleton will be extended and modified for specific job
configurations.
"""

from typing import Dict


def get_base_job() -> Dict:
    """Return the GitLab CI job body for a common test case. Include all default values."""
    return {
        "variables": {"ALPAKA_CI_OS_NAME": "Linux", "alpaka_CI": "GITLAB"},
        "script": ['echo "Hello World"'],
        "interruptible": True,
    }


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
