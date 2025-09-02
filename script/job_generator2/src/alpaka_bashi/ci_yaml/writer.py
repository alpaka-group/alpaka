"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Functionality to generate GitLab CI yaml from a combination and write it to an output.
"""

from typing import TextIO, Dict, Any
from typeguard import typechecked
import yaml
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.globals
from alpaka_bashi.globals import (
    JOB_EXECUTION_TYPE,
    CI_PIPELINE_NAME,
)
from alpaka_bashi.ci_yaml.names import get_job_name
from alpaka_bashi.ci_yaml.base_jobs import get_dummy_job


@typechecked
def get_job_yaml(
    combination_list: bashi.CombinationList, wave_sizes: Dict[ValueVersion, int] | None = None
) -> Dict[str, Any]:
    """Generate for each combination a GitLab CI yaml.

    Args:
        combination_list (bashi.CombinationList): combination-list
        wave_sizes (Dict[ValueVersion, int] | None, optional): The wave size defines how many jobs
        can be in one stage of a CI pipeline. The key defines the pipeline and value maximum number
        of jobs in a CI stage. If a pipeline is not defined in the dict, put all jobs in the same
        stage. Defaults to None.

    Returns:
        Dict[str, Any]: GitLab CI job yaml's
    """
    jobs: Dict[str, Any] = {"stages": []}

    stage_job_counter: Dict[ValueVersion, int] = {}
    if wave_sizes is not None:
        for wave_ver in wave_sizes:
            stage_job_counter[wave_ver] = 0

    for comb in combination_list:
        job_name = get_job_name(comb)
        wave_ver = comb[CI_PIPELINE_NAME].version
        stage_name = alpaka_bashi.globals.get_version_aliases()[CI_PIPELINE_NAME][wave_ver]

        if wave_sizes is not None and wave_ver in stage_job_counter:
            # dived number of already generated jobs by the wave size and round down.
            stage_name += f"_stage{int(stage_job_counter[wave_ver]/wave_sizes[wave_ver])}"
            stage_job_counter[wave_ver] += 1

        if stage_name not in jobs["stages"]:
            jobs["stages"].append(stage_name)

        job_body = {
            "stage": stage_name,
            "variables": {
                "JOB_EXECUTION_TYPE": alpaka_bashi.globals.get_version_aliases()[
                    JOB_EXECUTION_TYPE
                ][comb[JOB_EXECUTION_TYPE].version],
            },
            "scripts": ['echo "Hello World"', True],
        }
        jobs[job_name] = job_body

    return jobs


@typechecked
def get_dummy_job_yaml(stage_name: str = "") -> Dict[str, Any]:
    """Generate a dummy job, which can never fail.

    Args:
        stage_name (str, optional): Set stage, if string is not empty. Defaults to "".

    Returns:
        Dict[str, Any]: CI job yaml.
    """
    dummy_job: Dict[str, Any] = {}
    if stage_name != "":
        dummy_job["stages"] = [stage_name]

    dummy_job |= get_dummy_job()
    if stage_name != "":
        dummy_job["dummy-job"]["stage"] = stage_name
    return dummy_job


@typechecked
def write_job_yaml(
    jobs: Dict[str, Any],
    output_stream: TextIO,
):
    """Write Python data structure to yaml output.

    Args:
        jobs (Dict[str, Any]): GitLab CI jobs.
        output_stream (TextIO): Python output stream where yaml is written to. For example stdout or
        a file handle.
    """
    for key, body in jobs.items():
        yaml.dump({key: body}, output_stream)
        output_stream.write("\n")
