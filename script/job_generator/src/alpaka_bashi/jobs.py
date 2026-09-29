"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate GitLab CI jobs for a given combination
"""

import math
import re
from dataclasses import dataclass
from typing import Any

import bashi
import packaging.version
from bashi.globals import CLANG, GCC
from typeguard import typechecked

from alpaka_bashi.ci_yaml.names import get_job_name
from alpaka_bashi.globals import CI_PIPELINE_NAME, get_version_aliases
from alpaka_bashi.jobs_builder.clang_tidy import get_clang_tidy_job
from alpaka_bashi.jobs_builder.default import construct_job_yaml
from alpaka_bashi.jobs_builder.dummy import get_dummy_job
from alpaka_bashi.jobs_builder.emulated_simd import get_emulated_simd_job
from alpaka_bashi.jobs_builder.sanitizer import SanitizerType, get_sanitizer_job
from alpaka_bashi.versions import get_used_compiler_versions


@dataclass
class WaveSize:
    """Size of a wave. Depending the total number of jobs per wave, the size can be between size and size + extension.

    Members:
        - size (int): minium size of a wave
        - extension (int): the maximum number
    """

    size: int
    extension: int


def get_final_wave_sizes(
    combination_list: bashi.CombinationList, wave_sizes: dict[bashi.ValueVersion, WaveSize] | None = None
) -> dict[bashi.ValueVersion, int]:
    """Calculate the finale size of each wave. The finale size is between WaveSize.size and
    WaveSize.size + WaveSize.extension. If using size + extension removes the last stage,
    distribute all jobs evenly over the remaining stages. Therefore the function calculates the new
    wave size.
    """

    final_wave_sizes: dict[bashi.ValueVersion, int] = {}
    if wave_sizes is None:
        return final_wave_sizes

    for wave_ver in wave_sizes:
        final_wave_sizes[wave_ver] = 0

    for comb in combination_list:
        if comb[CI_PIPELINE_NAME].version in wave_sizes:
            final_wave_sizes[comb[CI_PIPELINE_NAME].version] += 1

    for wave_ver, number in final_wave_sizes.items():
        number_stages = math.ceil(number / wave_sizes[wave_ver].size)
        number_of_jobs_in_last_stage = number % wave_sizes[wave_ver].size

        final_wave_sizes[wave_ver] = wave_sizes[wave_ver].size
        number_reduced_stages = number_stages - 1
        number_of_extension_places = number_reduced_stages * wave_sizes[wave_ver].extension

        if 0 < number_of_jobs_in_last_stage < number_of_extension_places:
            required_places_per_wave = math.ceil(number_of_jobs_in_last_stage / number_reduced_stages)
            final_wave_sizes[wave_ver] += required_places_per_wave

    return final_wave_sizes


@typechecked
def get_job_configuration(
    combination_list: bashi.CombinationList,
    container_version: str,
    image_check: bool,
    stages: bool,
    wave_sizes: dict[bashi.ValueVersion, WaveSize] | None = None,
) -> dict[str, Any]:
    """Generate for each combination a GitLab CI yaml.

    Args:
        combination_list (bashi.CombinationList): combination-list
        container_version (str): Alpaka CI container tag.
        image_check (bool): If true, check if alpaka CI image exist (requires internet connection).
        stages (bool): If true, add stages.
        wave_sizes (Dict[ValueVersion, int] | None, optional): The wave size defines how many jobs
        can be in one stage of a CI pipeline. The key defines the pipeline and value maximum number
        of jobs in a CI stage. If a pipeline is not defined in the dict, put all jobs in the same
        stage. Defaults to None.

    Returns:
        Dict[str, Any]: GitLab CI job yaml's
    """
    jobs: dict[str, Any] = {}

    if len(combination_list) > 0 and stages:
        jobs["stages"] = []

    stage_job_counter: dict[bashi.ValueVersion, int] = {}
    final_wave_sizes: dict[bashi.ValueVersion, int] = get_final_wave_sizes(combination_list, wave_sizes)

    if wave_sizes is not None:
        for wave_ver in wave_sizes:
            stage_job_counter[wave_ver] = 0

    for comb in combination_list:
        job_name = get_job_name(comb)
        wave_ver = comb[CI_PIPELINE_NAME].version

        if stages:
            stage_name = get_version_aliases()[CI_PIPELINE_NAME][wave_ver]

            if wave_sizes is not None and wave_ver in stage_job_counter:
                # dived number of already generated jobs by the wave size and round down.
                stage_name += f"_stage{int(stage_job_counter[wave_ver] / final_wave_sizes[wave_ver])}"
                stage_job_counter[wave_ver] += 1

            if stage_name not in jobs["stages"]:
                jobs["stages"].append(stage_name)
        else:
            stage_name = ""

        jobs[job_name] = construct_job_yaml(comb, stage_name, container_version, image_check)

    return jobs


@typechecked
def get_special_jobs(
    container_version: str,
    image_check: bool,
    stage_name: str,
    job_filter: str,
) -> dict[str, Any]:
    """Return Dict of special CI jobs.

    Args:
        container_version (str): Container version.
        image_check (bool): Check if configured image exist. If not, use fallback.
        stage_name (str): Stage name. If empty, do not create stage property.
        job_filter (str): Filter jobs by job name. If empty, do not filter.

    Returns:
        Dict[str, Any]: Dict of CI jobs.
    """
    special_jobs: dict[str, Any] = {}

    if stage_name:
        special_jobs["stages"] = [stage_name]

    for compiler in (GCC, CLANG):
        for sanitzer in SanitizerType:
            special_jobs |= get_sanitizer_job(
                compiler_name=compiler,
                compiler_version=packaging.version.parse(str(max(get_used_compiler_versions()[compiler]))),
                sanitizer_type=sanitzer,
                container_version=container_version,
                stage_name=stage_name,
                image_check=image_check,
            )

    for compiler in (GCC, CLANG):
        special_jobs |= get_emulated_simd_job(
            compiler_name=compiler,
            compiler_version=packaging.version.parse(str(max(get_used_compiler_versions()[compiler]))),
            container_version=container_version,
            stage_name=stage_name,
            image_check=image_check,
        )

    special_jobs |= get_clang_tidy_job(
        compiler_version=packaging.version.parse("23"),
        container_version=container_version,
        stage_name=stage_name,
        image_check=image_check,
    )

    if job_filter:
        compiled_regex = re.compile(job_filter)
        special_jobs = {
            job_name: job_body
            for job_name, job_body in special_jobs.items()
            if compiled_regex.match(job_name) or job_name == "stages"
        }

    if len(special_jobs) == 0 or (len(special_jobs) == 1 and "stages" in special_jobs):
        special_jobs = get_dummy_job_yaml(stage_name)

    return special_jobs


@typechecked
def get_dummy_job_yaml(stage_name: str = "") -> dict[str, Any]:
    """Generate a dummy job, which can never fail.

    Args:
        stage_name (str, optional): Set stage, if string is not empty. Defaults to "".

    Returns:
        Dict[str, Any]: CI job yaml.
    """
    dummy_job: dict[str, Any] = {}
    if stage_name != "":
        dummy_job["stages"] = [stage_name]

    dummy_job |= get_dummy_job()
    if stage_name != "":
        dummy_job["dummy-job"]["stage"] = stage_name
    return dummy_job
