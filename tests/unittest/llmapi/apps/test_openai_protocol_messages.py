# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChatCompletionRequest rejects an empty `messages` array.

OpenAI returns 400 "Invalid 'messages': empty array"; without a validator the
request reached template rendering (a 200 with unprompted output on templates
that tolerate no turns, a template error otherwise) or an IndexError on
messages[-1] when add_generation_prompt is False.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("extra", [{}, {"add_generation_prompt": False}])
def test_empty_messages_rejected(extra):
    with pytest.raises(ValidationError, match="at least one message"):
        ChatCompletionRequest(model="m", messages=[], **extra)


def test_single_message_accepted():
    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert len(request.messages) == 1
