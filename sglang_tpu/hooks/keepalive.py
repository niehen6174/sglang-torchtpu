# SPDX-License-Identifier: Apache-2.0
"""TPU keep-alive while the scheduler idles (SGLANG_TPU_KEEPALIVE_SECONDS).

Why: tpuq reclaims a node after 15 minutes without TPU utilization, which an
     idle server (or one waiting between benchmark legs) reaches; no other
     process can keep the chips busy while the server holds them.
How: AFTER Scheduler.on_idle; tpu_keepalive.keepalive_tick runs a short
     matmul burst on the scheduler thread when no forward started for a while.
     Off unless the env sets a period.
Upstream PR or issue: none; a deployment concern of shared TPU pools.
Remove when: the pool stops reclaiming busy reservations, or never (opt-in).
"""

from sglang_tpu.tpu_keepalive import keepalive_after_idle

__all__ = ["keepalive_after_idle"]
