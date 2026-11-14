# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Result saving in ``trtllm-bench visual-gen``."""

import json

import pytest

from tensorrt_llm.bench.benchmark.visual_gen import _save_results
from tensorrt_llm.bench.benchmark.visual_gen_utils import VisualGenRequestOutput, calculate_metrics

pytestmark = pytest.mark.cpu_only

SUMMARY_KEYS = (
    "backend",
    "model",
    "duration",
    "total_requests",
    "completed",
    "request_throughput",
    "mean_latency",
    "percentiles_latency",
    "mean_generation",
    "percentiles_generation",
    "gen_params",
    "num_prompts",
    "max_concurrency",
    "num_gpus",
)


def _saved_result(tmp_path, *, save_detailed):
    outputs = [
        VisualGenRequestOutput(success=True, latency=2.0, generation=1.5),
        VisualGenRequestOutput(
            success=False, latency=0.5, error="boom", exception_type="RuntimeError"
        ),
    ]
    metrics = calculate_metrics(outputs, dur_s=4.0, selected_percentiles=[50.0], num_gpus=1)
    _save_results(
        backend="offline",
        model_id="org/model",
        benchmark_duration=4.0,
        metrics=metrics,
        outputs=outputs,
        gen_params={"num_inference_steps": 4},
        num_prompts=len(outputs),
        max_concurrency=1,
        num_gpus=1,
        save_detailed=save_detailed,
        result_dir=str(tmp_path),
        result_filename="result.json",
    )
    return json.loads((tmp_path / "result.json").read_text())


def test_save_result_without_save_detailed_drops_per_request_arrays(tmp_path):
    result = _saved_result(tmp_path, save_detailed=False)

    for key in ("latencies", "generations", "errors"):
        assert key not in result
    # Guard against a future per-request key slipping past the filter.
    assert [key for key, value in result.items() if isinstance(value, list)] == []
    for key in SUMMARY_KEYS:
        assert key in result
    assert result["completed"] == 1
    assert result["mean_latency"] == pytest.approx(2.0)


def test_save_result_with_save_detailed_keeps_per_request_arrays(tmp_path):
    result = _saved_result(tmp_path, save_detailed=True)

    assert result["latencies"] == [2.0, 0.5]
    assert result["generations"] == [1.5, 0.0]
    assert result["errors"] == ["", "boom"]
    for key in SUMMARY_KEYS:
        assert key in result
