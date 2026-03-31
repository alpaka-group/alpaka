"""Add several, special (, handwritten) CI jobs."""

import re
from typing import Dict, Any
from typeguard import typechecked
from .clang_analysis import get_clang_debug_analysis_job


@typechecked
def get_special_jobs(
    container_version: str,
    image_check: bool,
    stage_name: str,
    job_filter: str,
) -> Dict[str, Any]:
    """Return Dict of special CI jobs.

    Args:
        container_version (str): Container version.
        image_check (bool): Check if configured image exist. If not, use fallback.
        stage_name (str): Stage name. If empty, do not create stage property.
        job_filter (str): Filter jobs by job name. If empty, do not filter.

    Returns:
        Dict[str, Any]: Dict of CI jobs.
    """
    special_jobs: Dict[str, Any] = {}

    if stage_name:
        special_jobs["stages"] = [stage_name]

    special_jobs |= get_clang_debug_analysis_job(
        "14", "3.25.3", container_version, stage_name, image_check
    )

    if job_filter:
        compiled_regex = re.compile(job_filter)
        special_jobs = {
            job_name: job_body
            for job_name, job_body in special_jobs.items()
            if compiled_regex.match(job_name) or job_name == "stages"
        }

    return special_jobs
