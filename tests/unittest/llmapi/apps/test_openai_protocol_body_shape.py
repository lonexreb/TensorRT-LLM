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
"""OpenAI request models reject malformed bodies with a ValidationError.

Several ``mode="before"`` model validators called ``data.get(...)`` on the raw
body. A JSON body that is not an object (``[1, 2]``, ``"x"``) raised
AttributeError, and a string ``logprobs`` raised TypeError in the limit check.
Pydantic does not wrap those, so the server answered 500 instead of a 4xx
validation error.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)

pytestmark = pytest.mark.cpu_only

_MINIMAL_PAYLOADS = {
    CompletionRequest: {"model": "test-model", "prompt": "hi"},
    ChatCompletionRequest: {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    },
    ResponsesRequest: {"model": "test-model", "input": "hi"},
}


@pytest.mark.parametrize("request_cls", list(_MINIMAL_PAYLOADS))
def test_minimal_payload_validates(request_cls):
    request = request_cls.model_validate(dict(_MINIMAL_PAYLOADS[request_cls]))
    assert request.model == "test-model"


@pytest.mark.parametrize("body", [[1, 2], "x", 3, None])
@pytest.mark.parametrize("request_cls", list(_MINIMAL_PAYLOADS))
def test_non_object_body_raises_validation_error(request_cls, body):
    with pytest.raises(ValidationError) as exc_info:
        request_cls.model_validate(body)
    assert exc_info.value.errors()[0]["type"] == "model_type"


@pytest.mark.parametrize("body", ["[1, 2]", '"x"', "3", "null"])
@pytest.mark.parametrize("request_cls", list(_MINIMAL_PAYLOADS))
def test_non_object_json_body_raises_validation_error(request_cls, body):
    with pytest.raises(ValidationError):
        request_cls.model_validate_json(body)


def _completion(**extra):
    return CompletionRequest.model_validate({**_MINIMAL_PAYLOADS[CompletionRequest], **extra})


def _chat(**extra):
    return ChatCompletionRequest.model_validate(
        {**_MINIMAL_PAYLOADS[ChatCompletionRequest], **extra}
    )


def test_completion_logprobs_string_is_coerced():
    assert _completion(logprobs="5").logprobs == 5


@pytest.mark.parametrize("value", [MAX_TOP_LOGPROBS + 1, str(MAX_TOP_LOGPROBS + 1), -1])
def test_completion_logprobs_limit_applies_after_coercion(value):
    with pytest.raises(ValidationError, match="logprobs must be"):
        _completion(logprobs=value)


def test_completion_logprobs_at_limit_accepted():
    assert _completion(logprobs=MAX_TOP_LOGPROBS).logprobs == MAX_TOP_LOGPROBS


def test_chat_top_logprobs_string_is_coerced():
    assert _chat(logprobs=True, top_logprobs="5").top_logprobs == 5


@pytest.mark.parametrize("value", [MAX_TOP_LOGPROBS + 1, str(MAX_TOP_LOGPROBS + 1)])
def test_chat_top_logprobs_limit_applies_after_coercion(value):
    with pytest.raises(ValidationError, match="top_logprobs must be less than or equal"):
        _chat(logprobs=True, top_logprobs=value)


def test_chat_top_logprobs_requires_logprobs():
    with pytest.raises(ValidationError, match="logprobs must be true when using top_logprobs"):
        _chat(top_logprobs=2)


def test_chat_default_top_logprobs_does_not_require_logprobs():
    request = _chat(logprobs=False)
    assert request.top_logprobs == 0
    assert _chat(top_logprobs=None).top_logprobs is None


def test_object_body_rules_still_enforced():
    with pytest.raises(ValidationError, match="Stream options can only be defined"):
        _completion(stream_options={"include_usage": True})
    with pytest.raises(ValidationError, match="suffix is not supported"):
        _chat(suffix="tail")
    with pytest.raises(ValidationError, match="background can only be used"):
        ResponsesRequest.model_validate(
            {**_MINIMAL_PAYLOADS[ResponsesRequest], "background": True, "store": False}
        )
