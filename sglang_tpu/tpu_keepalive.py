# SPDX-License-Identifier: Apache-2.0
"""TPU keep-alive for a server that does no device work
(SGLANG_TPU_KEEPALIVE_SECONDS).

Shared TPU pools may reclaim a node that shows no TPU utilization for a while
(tpuq: 15 minutes), and a token op does not count: the load has to be a real
share of the time. A cold compile of a large model or a server waiting for
requests runs no device work, and no other process can fill in because the
server holds the chips.

keepalive_tick() runs short matmul bursts for SGLANG_TPU_KEEPALIVE_DUTY of the
time once no forward has started for SGLANG_TPU_KEEPALIVE_SECONDS. It runs on
the caller's thread, between the scheduler's idle iterations and between
precompile steps: torch_tpu's deferred execution is not thread-safe, and a
keep-alive thread failed forwards running at the same time.
"""

import time

import torch

# Longest a burst holds the caller (an idle scheduler takes no request meanwhile).
_BURST_S = 0.5

# time.monotonic() of the latest forward call (note_forward) and burst end.
_last_forward = 0.0
_last_burst_end = 0.0
_x: torch.Tensor | None = None


def note_forward() -> None:
    """Mark device work in progress; called at the start of every forward."""
    global _last_forward
    _last_forward = time.monotonic()


def keepalive_tick() -> None:
    """Run one burst if the server has been without device work long enough."""
    global _last_burst_end, _x
    from sglang_tpu.envs import envs as tpu_envs

    period = tpu_envs.SGLANG_TPU_KEEPALIVE_SECONDS.get()
    if period <= 0:
        return
    duty = tpu_envs.SGLANG_TPU_KEEPALIVE_DUTY.get()
    now = time.monotonic()
    if now - _last_forward < period:
        return
    if now - _last_burst_end < _BURST_S * (1.0 / duty - 1.0):
        return
    if _x is None:
        _x = torch.full((4096, 4096), 1e-3, dtype=torch.bfloat16, device="tpu")
    end = now + _BURST_S
    while time.monotonic() < end:
        y = _x
        for _ in range(16):
            y = y @ _x
        y.sum().item()  # wait for the device, so the duty is real time
    _last_burst_end = time.monotonic()


def keepalive_after_idle(result, scheduler):
    """AFTER hook for Scheduler.on_idle."""
    keepalive_tick()
    return result
