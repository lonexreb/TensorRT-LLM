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

import asyncio
import threading
from queue import Queue

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import ErrorResponse

pytestmark = pytest.mark.cpu_only


class _WorkerBaseStub:
    def await_responses(self, timeout):
        self.await_responses_timeout = timeout
        return self.engine_responses


class _RpcWorkerStub(RpcWorkerMixin, _WorkerBaseStub):
    def __init__(self, engine_responses):
        self.engine_responses = engine_responses
        self.rank = 0
        self._fetch_timeout = 0.1
        self._response_queue = Queue()
        self.enable_postprocess_parallel = False
        self._await_response_helper = AwaitResponseHelper(self)
        self._await_response_helper.responses_handler = self._responses_handler
        self.handler_responses = None

    def _responses_handler(self, responses):
        self.handler_responses = responses
        if responses:
            self._response_queue.put(responses)


@pytest.mark.parametrize(
    "engine_responses, expected_engine_responses",
    [
        (["first-response", "second-response"], ["first-response", "second-response"]),
        (
            [None, "first-response", None, "second-response", None],
            ["first-response", "second-response"],
        ),
        ([None, None], []),
        ([], []),
    ],
    ids=["normal", "mixed-none", "all-none", "empty"],
)
def test_fetch_responses_dispatches_engine_responses_and_submission_errors(
    engine_responses, expected_engine_responses
):
    worker = _RpcWorkerStub(engine_responses)
    original_responses = list(engine_responses)
    submission_error = ErrorResponse(client_id=1, error_msg="submit failed", request_id=2)
    worker._await_response_helper.temp_error_responses.put(submission_error)

    responses = worker.fetch_responses(timeout=0.25)

    assert worker.await_responses_timeout == 0.25
    expected = [*expected_engine_responses, submission_error]
    assert worker.handler_responses == expected
    assert responses == expected
    assert responses[-1] is submission_error
    assert engine_responses == original_responses
    assert worker._await_response_helper.temp_error_responses.empty()


class _CrashedEngineStub:
    """What PyExecutor leaves behind after its event loop dies."""

    def __init__(self, error):
        self._event_loop_error = error


def _crashed_worker(error):
    worker = _RpcWorkerStub([])
    worker.engine = _CrashedEngineStub(error)
    worker._results = {7: object(), 9: object()}
    worker.shutdown_event = threading.Event()
    return worker


def test_fetch_responses_reports_event_loop_crash_once():
    """Pending clients get one ErrorResponse each; the crash is reported once."""
    worker = _crashed_worker(RuntimeError("boom"))

    responses = worker.fetch_responses(timeout=0.1)

    assert [r.client_id for r in responses] == [7, 9]
    assert all(isinstance(r, ErrorResponse) for r in responses)
    assert all("boom" in r.error_msg for r in responses)
    assert worker._event_loop_error_reported
    # Nothing new on later polls: the clients already hold the error.
    assert worker.fetch_responses(timeout=0.1) == []


def test_fetch_responses_without_crash_reports_nothing():
    worker = _RpcWorkerStub([])
    worker.engine = _CrashedEngineStub(None)
    worker._results = {7: object()}

    assert worker.fetch_responses(timeout=0.1) == []
    assert not getattr(worker, "_event_loop_error_reported", False)


def test_fetch_responses_loop_ends_with_error_after_crash():
    """The stream carries the crash batch, then fails instead of spinning."""
    worker = _crashed_worker(RuntimeError("boom"))

    async def run():
        stream = worker.fetch_responses_loop_async()
        batch = await stream.__anext__()
        assert [r.client_id for r in batch] == [7, 9]
        with pytest.raises(RuntimeError, match="boom"):
            await stream.__anext__()

    asyncio.run(run())
