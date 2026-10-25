# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iterating a proxy stats batch must not wait a second `timeout`.

Both executor proxies fetch the whole batch over RPC (which already waits up
to `timeout`), put it on the IterationResult queue and hand the result out.
Nothing else feeds that queue, so iteration has to stop as soon as it drains
instead of waiting another full `timeout` (GitHub issue #19623: every
/metrics call paid a fixed ~2 s on top of the RPC).
"""

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.result import IterationResult
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy

pytestmark = pytest.mark.cpu_only

# Comfortably above a scheduler hiccup, far below the 2 s a regression costs.
MAX_DRAIN_SECONDS = 0.5
STATS = [{"iter": 0, "numActiveRequests": 1}, {"iter": 1, "numActiveRequests": 0}]


async def _drain(result: IterationResult) -> list:
    return [item async for item in result]


def test_zero_timeout_returns_queued_items_without_waiting():
    async def run():
        result = IterationResult()
        for stat in STATS:
            result.queue.put(json.dumps(stat))
        result.set_timeout(0)
        start = time.monotonic()
        items = await _drain(result)
        assert time.monotonic() - start < MAX_DRAIN_SECONDS
        assert items == STATS
        # Drained: the next iteration stops at once instead of raising QueueEmpty.
        assert await _drain(result) == []

    asyncio.run(run())


def test_zero_timeout_on_empty_queue_stops_cleanly():
    async def run():
        result = IterationResult()
        result.set_timeout(0)
        assert await _drain(result) == []
        # New prompts re-arm the result; an empty queue still ends iteration.
        result.mark_undone()
        assert await _drain(result) == []

    asyncio.run(run())


def _fake_proxy(cls, stats: list, events: list):
    """A proxy instance with only the attributes aget_stats/aget_kv_events use."""
    proxy = object.__new__(cls)
    proxy._is_llm_executor = True
    proxy._iter_stats_result = None
    proxy._iter_kv_events_result = None
    proxy.rpc_client = MagicMock()
    proxy.rpc_client.fetch_stats_wait_async.return_value.remote.return_value = [
        json.dumps(s) for s in stats
    ]
    proxy.rpc_client.fetch_kv_cache_events_wait_async.return_value.remote.return_value = [
        json.dumps(e) for e in events
    ]
    return proxy


@pytest.mark.parametrize(
    "cls", [GenerationExecutorProxy, GenerationExecutorRpcProxy], ids=["ipc_proxy", "rpc_proxy"]
)
def test_proxy_stats_batch_drains_without_second_wait(cls):
    events = [{"event_id": 7}]

    async def run():
        proxy = _fake_proxy(cls, STATS, events)
        timeout = 2.0

        start = time.monotonic()
        stats = await _drain(proxy.aget_stats(timeout))
        kv_events = await _drain(proxy.aget_kv_events(timeout))
        elapsed = time.monotonic() - start

        assert stats == STATS
        assert kv_events == events
        assert elapsed < MAX_DRAIN_SECONDS, (
            f"draining took {elapsed:.2f}s: iteration waited `timeout` again"
        )
        # `timeout` still bounds the RPC wait itself.
        proxy.rpc_client.fetch_stats_wait_async.assert_called_once_with(timeout=timeout)
        proxy.rpc_client.fetch_kv_cache_events_wait_async.assert_called_once_with(timeout=timeout)

        # A later call hands out a fresh batch, not a stale "done" result.
        proxy.rpc_client.fetch_stats_wait_async.return_value.remote.return_value = [
            json.dumps({"iter": 2})
        ]
        assert await _drain(proxy.aget_stats(timeout)) == [{"iter": 2}]

    asyncio.run(run())
