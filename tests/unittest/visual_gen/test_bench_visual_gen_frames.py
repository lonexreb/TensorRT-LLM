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
"""``trtllm-bench visual-gen`` turns ``--seconds`` into the frame count it generates."""

import json
from types import SimpleNamespace
from unittest import mock

import click
import pytest
from click.testing import CliRunner

import tensorrt_llm.visual_gen as visual_gen_api
from tensorrt_llm.bench.benchmark import visual_gen as bench_visual_gen

pytestmark = pytest.mark.cpu_only


def test_seconds_converts_to_frames_like_the_server():
    assert bench_visual_gen._resolve_num_frames(None, 8.0, 16) == (128, 8.0)


def test_seconds_conversion_truncates_like_the_server():
    assert bench_visual_gen._resolve_num_frames(None, 4.1, 16) == (65, 4.1)


def test_num_frames_wins_and_seconds_is_derived_from_it():
    assert bench_visual_gen._resolve_num_frames(81, 8.0, 16) == (81, 81 / 16)


def test_neither_given_leaves_the_model_default():
    assert bench_visual_gen._resolve_num_frames(None, None, 16) == (None, None)


@pytest.mark.parametrize("seconds", [0.0, 0.05, -1.0])
def test_seconds_below_one_frame_is_rejected(seconds):
    with pytest.raises(click.BadParameter, match="at least 1 frame is required"):
        bench_visual_gen._resolve_num_frames(None, seconds, 16)


def _run_cli(tmp_path, monkeypatch, *extra_args):
    """Run the command with VisualGen stubbed; return (params sent, report gen_params)."""
    engine = mock.MagicMock()
    engine.generate.return_value = SimpleNamespace(image=None, video=None, metrics=None)
    monkeypatch.setattr(visual_gen_api, "VisualGen", mock.MagicMock(return_value=engine))
    monkeypatch.setattr(bench_visual_gen, "_mark_visual_gen", lambda: None)

    result = CliRunner().invoke(
        bench_visual_gen.visual_gen_command,
        [
            "--prompt",
            "a cat",
            "--num_prompts",
            "1",
            "--warmup",
            "0",
            "--save_result",
            "--result_dir",
            str(tmp_path),
            "--result_filename",
            "result.json",
            *extra_args,
        ],
        obj=SimpleNamespace(model="dummy-model", checkpoint_path=None),
    )
    assert result.exit_code == 0, result.output
    params = engine.generate.call_args.kwargs["params"]
    report = json.loads((tmp_path / "result.json").read_text())["gen_params"]
    return params, report


def test_cli_seconds_reaches_the_pipeline_and_the_report(tmp_path, monkeypatch):
    params, report = _run_cli(tmp_path, monkeypatch, "--seconds", "8", "--fps", "16")
    assert params.num_frames == 128
    assert report["seconds"] == 8.0
    assert report["num_frames"] == 128


def test_cli_without_seconds_or_num_frames_keeps_the_model_default(tmp_path, monkeypatch):
    params, report = _run_cli(tmp_path, monkeypatch)
    assert params.num_frames is None
    assert report["seconds"] is None
    assert report["num_frames"] is None


def test_cli_rejects_seconds_below_one_frame(monkeypatch):
    factory = mock.MagicMock()
    monkeypatch.setattr(visual_gen_api, "VisualGen", factory)
    monkeypatch.setattr(bench_visual_gen, "_mark_visual_gen", lambda: None)

    result = CliRunner().invoke(
        bench_visual_gen.visual_gen_command,
        ["--prompt", "a cat", "--seconds", "0.01", "--fps", "16"],
        obj=SimpleNamespace(model="dummy-model", checkpoint_path=None),
    )
    assert result.exit_code == 2
    assert "at least 1 frame is required" in result.output
    factory.assert_not_called()
