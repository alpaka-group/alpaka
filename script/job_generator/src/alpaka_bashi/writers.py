"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Write combination to stdout or file
"""

import argparse
from itertools import chain
from typing import Any, TextIO

import bashi
import yaml
from typeguard import typechecked

from alpaka_bashi.globals import CI_PIPELINE_NAME, CI_PIPELINE_SPACK, CI_PIPELINE_SPECIAL, get_version_aliases
from alpaka_bashi.jobs import (
    WaveSize,
    get_dummy_job_yaml,
    get_job_configuration,
    get_spack_build_jobs,
    get_special_jobs,
)
from alpaka_bashi.utils import get_filter_name


@typechecked
def write_job_yaml(
    jobs: dict[str, Any],
    output_stream: TextIO,
):
    """Write Python data structure to yaml output.

    Args:
        jobs (Dict[str, Any]): GitLab CI jobs.
        output_stream (TextIO): Python output stream where yaml is written to. For example stdout or
        a file handle.
    """
    for key, body in jobs.items():
        yaml.dump({key: body}, output_stream, Dumper=bashi.MultilineDumper)
        output_stream.write("\n")


def write_single_file_job_configuration(
    pipelines: dict[bashi.ValueVersion, bashi.CombinationList],
    spack_builder: bashi.SpackBuilder,
    spack_package_repository_sha: str,
    args: argparse.Namespace,
    output_stream: TextIO,
):
    """Write generated GitLab CI yaml code to stdout.

    Args:
        pipelines (dict[bashi.ValueVersion, bashi.CombinationList]): All CI pipelines and their jobs.
        spack_builder (bashi.SpackBuilder): Contains all spack packages, which needs to be build and additional
            information.
        spack_package_repository_sha (str): Set spack package repository sha, which should be
            used for the spack package build jobs. If empty, do not update repository.
        args (argparse.Namespace): Application arguments.
        output_stream (TextIO): Output stream for the yaml code.
    """
    job_filter_name = get_filter_name(args)

    jobs: dict[str, Any] = {}
    # Stages are mandatory in single file, if spack packages needs to be build.
    # The spack packages needs to be available before the test jobs are started.
    single_file_stages = []

    if spack_builder.get_build_cache_packages().missing_packages:
        single_file_stages.append(CI_PIPELINE_SPACK)
        jobs |= get_spack_build_jobs(spack_builder, spack_package_repository_sha, CI_PIPELINE_SPACK)

    jobs |= get_job_configuration(
        combination_list=list(chain(*pipelines.values())),
        container_version=str(args.version),
        image_check=args.no_image_check,
        stages=len(single_file_stages) > 0,
        wave_sizes=None,
    )

    # If no spack packages needs to be build, no build spack package build was created.
    # Therefore other stages are also not required.
    if single_file_stages:
        single_file_stages += jobs["stages"]

    jobs |= get_special_jobs(
        container_version=str(args.version),
        image_check=args.no_image_check,
        stage_name=CI_PIPELINE_SPECIAL if len(single_file_stages) > 0 else "",
        job_filter=job_filter_name,
    )

    if single_file_stages:
        single_file_stages += jobs["stages"]

    if single_file_stages:
        jobs["stages"] = single_file_stages

    if len(jobs) == 0:
        jobs = get_dummy_job_yaml()

    write_job_yaml(jobs, output_stream)


def write_multiple_file_job_configuration(
    pipelines: dict[bashi.ValueVersion, bashi.CombinationList],
    spack_builder: bashi.SpackBuilder,
    spack_package_repository_sha: str,
    wave_sizes: dict[bashi.ValueVersion, WaveSize],
    args: argparse.Namespace,
):
    """Write generated GitLab CI yaml code to different files.

    Args:
        pipelines (dict[bashi.ValueVersion, bashi.CombinationList]): All CI pipelines and their
        jobs.
        spack_builder (bashi.SpackBuilder): Contains all spack packages, which needs to be build and additional
            information.
        spack_package_repository_sha (str): Set spack package repository sha, which should be
            used for the spack package build jobs. If empty, do not update repository.
        wave_sizes (dict[bashi.ValueVersion, int]): Size of each wave.
        args (argparse.Namespace): Application arguments.
    """
    job_filter_name = get_filter_name(args)

    for pipeline_ver, combinations in pipelines.items():
        pipeline_name = get_version_aliases()[CI_PIPELINE_NAME][pipeline_ver]
        output_path = getattr(args, f"pipeline-out-{pipeline_name}".replace("-", "_"))

        if pipeline_name == CI_PIPELINE_SPECIAL:
            jobs = get_special_jobs(
                container_version=str(args.version),
                image_check=args.no_image_check,
                stage_name=pipeline_name,
                job_filter=job_filter_name,
            )
        elif pipeline_name == CI_PIPELINE_SPACK:
            jobs = get_spack_build_jobs(spack_builder, spack_package_repository_sha, pipeline_name)
        else:
            jobs = get_job_configuration(
                combination_list=combinations,
                container_version=str(args.version),
                image_check=args.no_image_check,
                stages=True,
                wave_sizes=wave_sizes,
            )

        if len(jobs) == 0:
            jobs = get_dummy_job_yaml(pipeline_name)

        with open(output_path, "w", encoding="utf-8") as output_file:
            write_job_yaml(jobs, output_file)
