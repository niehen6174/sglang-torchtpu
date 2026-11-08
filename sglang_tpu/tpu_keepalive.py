# SPDX-License-Identifier: Apache-2.0
"""TPU keep-alive inside a server process while it does no device work
(SGLANG_TPU_KEEPALIVE_SECONDS).

Shared TPU pools may reclaim a node that shows no TPU utilization for a while
(tpuq: 15 minutes), and a token op does not count: the load has to be a real
share of each period. A cold compile of a large model or a server waiting for
requests runs no device work, and no other process can fill in because the
server holds the chips. While no forward has started for one period, this
daemon thread runs matmuls for SGLANG_TPU_KEEPALIVE_DUTY of each period; once
forwards run again it stays out of their way.
"""

import logging
import threading
import time

import torch

logger = logging.getLogger(__name__)

_started = False
# time.monotonic() of the latest forward call (note_forward).
_last_forward = 0.0


def note_forward() -> None:
    """Mark device work in progress; called at the start of every forward."""
    global _last_forward
    _last_forward = time.monotonic()


def _burst(x: torch.Tensor, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        y = x
        for _ in range(16):
            y = y @ x
        y.sum().item()  # wait for the device, so the duty is real time


def _loop(period: float, duty: float) -> None:
    x = torch.full((4096, 4096), 1e-3, dtype=torch.bfloat16, device="tpu")
    while True:
        time.sleep(period * (1.0 - duty))
        if time.monotonic() - _last_forward < period:
            continue
        try:
            _burst(x, period * duty)
        except Exception:  # noqa: BLE001 - a failed burst must not take the server down
            logger.exception("TPU keep-alive burst failed; stopping it.")
            return


def maybe_start_keepalive() -> None:
    """Start the keep-alive thread once per process if the env asks for it."""
    global _started
    from sglang_tpu.envs import envs as tpu_envs

    period = tpu_envs.SGLANG_TPU_KEEPALIVE_SECONDS.get()
    duty = tpu_envs.SGLANG_TPU_KEEPALIVE_DUTY.get()
    if _started or period <= 0:
        return
    _started = True
    threading.Thread(
        target=_loop, args=(period, duty), name="tpu-keepalive", daemon=True
    ).start()
    logger.info(
        "TPU keep-alive: matmuls for %.0f%% of every %.0f s with no forward.",
        100 * duty,
        period,
    )
