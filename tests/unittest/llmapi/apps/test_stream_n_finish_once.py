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
"""Streaming with n > 1 must emit each choice's terminal chunk exactly once."""

import json

import pytest

from tensorrt_llm.executor import GenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    chat_stream_post_processor,
    completion_stream_post_processor,
)

# The CPU CI stage runs pytest with -m cpu_only and skips files without this marker.
pytestmark = pytest.mark.cpu_only


def _stream_payloads_from_chunks(chunks):
    payloads = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        for line in chunk.splitlines():
            if line.startswith("data: "):
                data = line[len("data: ") :].strip()
                if data != "[DONE]":
                    payloads.append(json.loads(data))
    return payloads


def _two_candidate_result():
    """n=2 result after one step: candidate 0 finished, candidate 1 not."""
    result = GenerationResultBase(
        7, SamplingParams(n=2, temperature=0.8, top_p=0.9))
    first, second = result._outputs
    first.text, first.token_ids, first.finish_reason = "a", [1], "stop"
    second.text, second.token_ids = "b", [2]
    return result


def _advance_second_candidate(result):
    """Simulate the executor delivering candidate 1's next step.

    Only the handled candidate's diff markers move (result._handle_sequence);
    candidate 0 keeps its stale text/token diffs exactly as on main.
    """
    second = result._outputs[1]
    second._last_text_len = len(second.text)
    second._last_token_ids_len = len(second.token_ids)
    second.text = "bc"
    second.token_ids.append(3)
    second.finish_reason = "stop"
    result._done = True


def _finish_counts(payloads):
    counts = {}
    for payload in payloads:
        for choice in payload["choices"]:
            if choice.get("finish_reason") is not None:
                counts[choice["index"]] = counts.get(choice["index"], 0) + 1
    return counts


def test_completion_stream_post_processor_emits_each_finish_once() -> None:
    result = _two_candidate_result()
    args = CompletionPostprocArgs(model="test-model", num_choices=2)

    first_call = completion_stream_post_processor(result, args)
    _advance_second_candidate(result)
    second_call = completion_stream_post_processor(result, args)

    assert _finish_counts(_stream_payloads_from_chunks(first_call)) == {0: 1}
    # Candidate 0 was already closed: its sibling's step must not re-emit it.
    second_indexes = [
        choice["index"] for payload in _stream_payloads_from_chunks(second_call)
        for choice in payload["choices"]
    ]
    assert second_indexes == [1]
    assert _finish_counts(_stream_payloads_from_chunks(first_call +
                                                       second_call)) == {
                                                           0: 1,
                                                           1: 1
                                                       }


def test_chat_stream_post_processor_emits_each_finish_once() -> None:
    result = _two_candidate_result()
    args = ChatPostprocArgs(role="assistant", model="test-model", num_choices=2)

    first_call = chat_stream_post_processor(result, args)
    _advance_second_candidate(result)
    second_call = chat_stream_post_processor(result, args)

    assert _finish_counts(_stream_payloads_from_chunks(first_call)) == {0: 1}
    second_indexes = [
        choice["index"] for payload in _stream_payloads_from_chunks(second_call)
        for choice in payload["choices"]
    ]
    assert second_indexes == [1]
    assert _finish_counts(_stream_payloads_from_chunks(first_call +
                                                       second_call)) == {
                                                           0: 1,
                                                           1: 1
                                                       }
