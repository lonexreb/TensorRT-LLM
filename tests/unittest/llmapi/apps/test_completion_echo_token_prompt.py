# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/v1/completions ``echo`` with a token-id prompt.

The server stores the raw prompt item on CompletionPostprocArgs.prompt, so a
token-id prompt arrives as a list; the post-processors used to concatenate
it with the generated text and raise ``TypeError: can only concatenate list
(not "str") to list`` (a 400 non-streaming, an SSE error event streaming).
"""
import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor import GenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.postprocess_handlers import (
    CompletionPostprocArgs,
    completion_response_post_processor,
    completion_stream_post_processor,
)

pytestmark = pytest.mark.cpu_only


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, token_ids, **kwargs):
        self.calls.append(list(token_ids))
        return "AB"


def _result(text="x", token_ids=(7,)):
    result = GenerationResultBase(1, SamplingParams())
    output = result._outputs[0]
    output.text = text
    output.token_ids = list(token_ids)
    output.finish_reason = "stop"
    result._done = True
    return result


def _args(tokenizer, prompt):
    return CompletionPostprocArgs(
        model="m", echo=True, prompt=prompt, num_prompt_tokens=2, tokenizer=tokenizer
    )


def test_echo_token_id_prompt_is_detokenized_non_streaming():
    tokenizer = _FakeTokenizer()

    response = completion_response_post_processor(_result(), _args(tokenizer, [1, 2]))

    assert response.choices[0].text == "ABx"
    assert tokenizer.calls == [[1, 2]]


def test_echo_token_id_prompt_is_detokenized_once_when_streaming():
    tokenizer = _FakeTokenizer()
    args = _args(tokenizer, [1, 2])
    result = _result()

    chunks = completion_stream_post_processor(result, args)
    args.first_iteration = False
    result._outputs[0]._last_text_len = len(result._outputs[0].text)
    result._outputs[0].text = "xy"
    chunks += completion_stream_post_processor(result, args)

    texts = [
        json.loads(chunk.removeprefix("data: "))["choices"][0]["text"]
        for chunk in chunks
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]")
    ]
    assert texts[0] == "ABx"
    assert "AB" not in "".join(texts[1:])
    assert tokenizer.calls == [[1, 2]]


def test_echo_text_prompt_unchanged():
    assert completion_response_post_processor(_result(), _args(None, "P")).choices[0].text == "Px"


def test_echo_token_id_prompt_without_tokenizer_is_a_clear_error():
    with pytest.raises(ValueError, match="requires a tokenizer"):
        completion_response_post_processor(_result(), _args(None, [1, 2]))
