# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An RPC worker crash must reach RpcExecutorMixin clients (no GPU).

Before, a failing response stream only ended the main-loop thread and a
worker future holding an exception was only read at shutdown(): nothing hit
_error_queue, check_health() stayed green and every pending result waited
forever. These tests bind the real mixin / proxy methods to stubs.
"""

import asyncio
import queue
import threading
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor import EngineDeadError
from tensorrt_llm.executor.rpc.rpc_common import RPCStreamingError
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin

pytestmark = pytest.mark.cpu_only


class _FakeResult:
    def __init__(self):
        self.queue = queue.Queue()


class _FailingStream:
    """remote_streaming() that delivers one batch, then breaks."""

    def __init__(self, error):
        self._error = error

    async def remote_streaming(self):
        yield [[]]
        raise self._error


class _FakeRpcClient:
    def __init__(self, error):
        self._error = error

    def fetch_responses_loop_async(self):
        return _FailingStream(self._error)


class _MixinStub(RpcExecutorMixin):
    def __init__(self, error):
        self.rpc_client = _FakeRpcClient(error)
        self._results = {1: _FakeResult(), 2: _FakeResult()}
        self._error_queue = queue.Queue()
        self._shutdown_event = threading.Event()


def test_broken_response_stream_fails_pending_results():
    error = RPCStreamingError("stream broke")
    stub = _MixinStub(error)

    with pytest.raises(RPCStreamingError):
        asyncio.run(stub._fetch_responses_loop_async())

    assert stub._error_queue.get_nowait() is error
    for result in stub._results.values():
        item = result.queue.get_nowait()
        assert isinstance(item, EngineDeadError)
        assert item.root_cause is error


def test_stream_error_during_shutdown_is_not_fatal():
    stub = _MixinStub(RPCStreamingError("cancelled"))
    stub._shutdown_event.set()

    with pytest.raises(RPCStreamingError):
        asyncio.run(stub._fetch_responses_loop_async())

    assert stub._error_queue.empty()
    assert all(r.queue.empty() for r in stub._results.values())


def _bare_proxy():
    proxy = GenerationExecutorRpcProxy.__new__(GenerationExecutorRpcProxy)
    proxy.rpc_addr = "ipc:///tmp/unused"
    proxy.worker_kwargs = {}
    proxy._results = {1: _FakeResult()}
    proxy._error_queue = queue.Queue()
    proxy._shutdown_event = threading.Event()
    return proxy


def test_worker_future_exception_fails_pending_results():
    """A rank that died in main_task is reported as soon as its future resolves."""
    proxy = _bare_proxy()
    healthy, dead = Future(), Future()
    healthy.set_result(None)
    cause = RuntimeError("rank 1 died")
    dead.set_exception(cause)
    proxy.mpi_session = Mock()
    proxy.mpi_session.submit.return_value = [healthy, dead]

    # add_done_callback runs synchronously on an already-completed future.
    proxy.launch_workers()

    assert proxy._error_queue.get_nowait() is cause
    item = proxy._results[1].queue.get_nowait()
    assert isinstance(item, EngineDeadError)
    assert item.root_cause is cause


def test_worker_future_success_is_quiet():
    proxy = _bare_proxy()
    done = Future()
    done.set_result(None)
    proxy.mpi_session = Mock()
    proxy.mpi_session.submit.return_value = [done]

    proxy.launch_workers()

    assert proxy._error_queue.empty()
    assert proxy._results[1].queue.empty()
