# SPDX-License-Identifier: Apache-2.0
"""Low-duty TPU keep-alive inside a server process (SGLANG_TPU_KEEPALIVE_SECONDS).

Shared TPU pools may reclaim a node that shows no TPU utilization for a while
(tpuq: 15 minutes). A cold compile of a large model, or a server idling between
benchmark legs, uses no TensorCore time, and no other process can keep the chips
busy because the server holds them. This daemon thread runs one small matmul on
the process's TPU device every period: about a millisecond per period.
"""

import logging
import threading

import torch

logger = logging.getLogger(__name__)

_started = False


def _loop(period: float) -> None:
    x = torch.ones((1024, 1024), dtype=torch.bfloat16, device="tpu")
    stop = threading.Event()
    while not stop.wait(period):
        try:
            (x @ x).sum().item()
        except Exception:  # noqa: BLE001 - a failed tick must not take the server down
            logger.exception("TPU keep-alive tick failed; stopping it.")
            return


def maybe_start_keepalive() -> None:
    """Start the keep-alive thread once per process if the env asks for it."""
    global _started
    from sglang_tpu.envs import envs as tpu_envs

    period = tpu_envs.SGLANG_TPU_KEEPALIVE_SECONDS.get()
    if _started or period <= 0:
        return
    _started = True
    threading.Thread(
        target=_loop, args=(period,), name="tpu-keepalive", daemon=True
    ).start()
    logger.info("TPU keep-alive: one 1024^2 matmul every %.0f s.", period)
