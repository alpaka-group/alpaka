"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generates the GitLab CI jobs for alpaka.
"""

import argparse
import random
import sys

import bashi

import alpaka_bashi
from alpaka_bashi.globals import CI_PIPELINE_COMPILE_ONLY_VER, CI_PIPELINE_NAME_MAPPING, CI_PIPELINE_RUNTIME_CPU_VER


def get_args() -> argparse.Namespace:
    """Define and parse the commandline arguments.

    Returns:
        argparse.Namespace: The commandline arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate job matrix and create GitLab CI .yml.")

    parser.add_argument("version", type=float, help="Version number of the used CI container.")
    parser.add_argument(
        "--print-combinations",
        action="store_true",
        help="Display combination list.",
    )

    parser.add_argument(
        "--no-image-check",
        action="store_false",
        help="Disable registry check for existing Docker image.",
    )

    parser.add_argument(
        "--no-verification",
        action="store_true",
        help="Disable verification of the combination and continue generating the GitLab CI yaml code.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter the jobs with a Python regex that checks the job names.",
    )

    parser.add_argument(
        "--split-pipeline",
        action="store_true",
        help="Write job pipelines in separate output files.",
    )

    for wave_name in CI_PIPELINE_NAME_MAPPING:
        parser.add_argument(
            f"--pipeline-out-{wave_name}",
            type=str,
            required="--split-pipeline" in sys.argv,
            # add `all` and remove `JOB_UNKNOWN` from the choices
            help=f"Output path of the job yaml for the pipeline {wave_name}",
        )

    parser.add_argument(
        "--debug-print",
        type=bashi.FilterDebugMode,
        choices=list(bashi.FilterDebugMode),
        default="off",
        help="Display Indicate which combinations passed through the filter chain and which did "
        "not.Green text indicates that the combination passed through the filter chain; red text"
        " indicates that it did not. Add the keyword `passed` if colored output is not available."
        "Option `normal` is easy human readable output. If `args` is set, the output can be "
        "directly passed to the validator",
    )

    return parser.parse_args()


def setup_row_printer() -> None:
    """Set extra configurations for the bashi.print_row_nice() function"""
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.BUILD_TYPE, "buildType")
    # bashi.add_print_row_nice_parameter_alias(alpaka_bashi.JOB_EXECUTION_TYPE, "jobType")

    for val_name, aliases in alpaka_bashi.get_version_aliases().items():
        bashi.add_print_row_nice_version_alias(val_name, aliases)


def main() -> None:
    """The main entry point."""
    args = get_args()

    setup_row_printer()

    software_versions = alpaka_bashi.get_software_versions_for_alpaka()
    param_matrix: bashi.ParameterValueMatrix = bashi.get_parameter_value_matrix(
        software_versions=software_versions, backends=alpaka_bashi.get_used_backends()
    )

    version_relation = alpaka_bashi.get_alpaka_version_relation()
    alpaka_filter = alpaka_bashi.AlpakaFilter()
    runtime_infos = bashi.get_runtime_infos(param_matrix, version_relation)

    comb_list: bashi.CombinationList = bashi.generate_combination_list(
        parameter_value_matrix=param_matrix,
        runtime_infos=runtime_infos,
        custom_filter=alpaka_filter,
        version_relation=version_relation,
        debug_print=args.debug_print,
    )
    print(f"number of combinations: {len(comb_list)}", file=sys.stderr)

    comb_list = alpaka_bashi.add_combinations_parameters(comb_list)

    if not args.no_verification:
        if not alpaka_bashi.verify(comb_list, param_matrix, version_relation, runtime_infos):
            print("ERROR: Result is incorrect", file=sys.stderr)
            sys.exit(1)
        else:
            print("Result is correct", file=sys.stderr)
    else:
        alpaka_bashi.print_warn("Skip verification step")

    job_filter_name = alpaka_bashi.get_filter_name(args)
    if job_filter_name:
        comb_list = alpaka_bashi.filter_combinations(comb_list, job_filter_name)
        print(f"number of filtered combinations: {len(comb_list)}", file=sys.stderr)

    if args.print_combinations:
        for c in comb_list:
            bashi.print_row_nice(c)
        sys.exit(0)

    # shuffle jobs to increase the chance to run different compiler in the first wave
    random.Random(42).shuffle(comb_list)

    pipelines = alpaka_bashi.distribute_to_pipelines(comb_list)

    # If the pipelines are not split and therefore written to different files, write everything
    # to stdout.
    # We split up the pipelines and merge again, because in the meantime reorder operations can be
    # applied on the different pipelines.
    # By the way, it also automatically sort the jobs by pipeline.
    if not args.split_pipeline:
        alpaka_bashi.write_single_file_job_configuration(pipelines, args, sys.stdout)
    else:
        wave_sizes = {
            CI_PIPELINE_COMPILE_ONLY_VER: alpaka_bashi.WaveSize(30, 2),
            CI_PIPELINE_RUNTIME_CPU_VER: alpaka_bashi.WaveSize(30, 2),
        }

        alpaka_bashi.write_multiple_file_job_configuration(pipelines, wave_sizes, args)

    sys.exit(0)


if __name__ == "__main__":
    main()
