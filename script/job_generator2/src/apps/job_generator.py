"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate CI jobs for alpaka.
"""

import sys
import argparse
import bashi
import alpaka_bashi


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
        "--filter",
        type=str,
        default="",
        help="Filter the jobs with a Python regex that checks the job names.",
    )

    parser.add_argument(
        "--reorder",
        type=str,
        default="",
        help="Orders jobs by their names. Expects a string consisting of one or more Python regex. "
        'The regex are separated by whitespaces. For example, the regex "^NVCC ^GCC" has the '
        "behavior that all NVCC jobs are executed first and then all GCC jobs.",
    )

    parser.add_argument(
        "--split-pipeline",
        action="store_true",
        help="Write job pipelines in separate output files.",
    )

    for wave_name in alpaka_bashi.globals.CI_PIPELINE_NAME_MAPPING:
        parser.add_argument(
            f"--pipeline-out-{wave_name}",
            type=str,
            required="--split-pipeline" in sys.argv,
            # add `all` and remove `JOB_UNKNOWN` from the choices
            help=f"Output path of the job yaml for the pipeline {wave_name}",
        )

    parser.add_argument(
        "--no-image-check",
        action="store_false",
        help="Disable registry check for existing Docker image.",
    )

    return parser.parse_args()


def setup_row_printer() -> None:
    """Set extra configurations for the bashi.print_row_nice() function"""
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.BUILD_TYPE, "buildType")
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.JOB_EXECUTION_TYPE, "jobType")

    for val_name, aliases in alpaka_bashi.get_version_aliases().items():
        bashi.add_print_row_nice_version_alias(val_name, aliases)


# pylint: disable=too-many-locals
def main() -> None:
    """Entry point"""
    args = get_args()

    setup_row_printer()

    param_matrix: bashi.ParameterValueMatrix = bashi.get_parameter_value_matrix(
        software_versions=alpaka_bashi.get_alpaka_version()
    )

    alpaka_filter = alpaka_bashi.AlpakaFilter()
    runtime_infos = bashi.get_runtime_infos(param_matrix)
    runtime_infos |= alpaka_bashi.get_runtime_infos()

    comb_list: bashi.CombinationList = bashi.generate_combination_list(
        parameter_value_matrix=param_matrix,
        runtime_infos=runtime_infos,
        custom_filter=alpaka_filter,
        # change me to display which combinations passed and did not pass the filter chain
        debug_print=bashi.FilterDebugMode.OFF,
    )

    print(f"number of combinations: {len(comb_list)}", file=sys.stderr)

    alpaka_bashi.add_combinations_parameters(comb_list)

    if not alpaka_bashi.verify(comb_list, param_matrix, runtime_infos):
        print("ERROR: Result is incorrect", file=sys.stderr)
        sys.exit(1)

    print("Result is correct", file=sys.stderr)

    if args.print_combinations:
        for c in comb_list:
            bashi.print_row_nice(c)
        sys.exit(0)

    pipelines = alpaka_bashi.distribute_to_pipelines(comb_list)

    wave_sizes = {alpaka_bashi.globals.CI_PIPELINE_COMPILE_ONLY_VER: 20}

    # If the pipelines are not splitted and therefore written to different files, write everything
    # to stdout.
    # We split up the pipelines and merge again, because in the meantime reorder operations can be
    # applied on the different pipelines.
    # By the way, it also automatically sort the jobs by pipeline.
    if not args.split_pipeline:
        single_pipeline: bashi.CombinationList = []
        for pipeline in pipelines.values():
            single_pipeline += pipeline
        jobs = alpaka_bashi.get_job_yaml(single_pipeline, wave_sizes)
        alpaka_bashi.write_job_yaml(jobs, sys.stdout)
    else:
        for pipeline_ver, combinations in pipelines.items():
            pipeline_name = alpaka_bashi.get_version_aliases()[
                alpaka_bashi.globals.CI_PIPELINE_NAME
            ][pipeline_ver]
            output_path = getattr(args, f"pipeline-out-{pipeline_name}".replace("-", "_"))

            if len(combinations) > 0:
                jobs = alpaka_bashi.get_job_yaml(combinations, wave_sizes)
            else:
                jobs = alpaka_bashi.get_dummy_job_yaml(pipeline_name)
            with open(output_path, "w", encoding="utf-8") as output_file:
                alpaka_bashi.write_job_yaml(jobs, output_file)

    sys.exit(0)


if __name__ == "__main__":
    main()
