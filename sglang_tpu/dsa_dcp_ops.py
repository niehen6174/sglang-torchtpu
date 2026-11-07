# SPDX-License-Identifier: Apache-2.0
"""Torch custom ops for sparse MLA (DSA) under decode context parallelism.

Under DCP each rank of a group holds the MLA latents and DSA index keys of the
positions it owns (page-striped, ``interleave_size`` tokens per run). One DSA
layer then runs, per rank:

1. ``dsa_index_topk_dcp``: store the owned index keys, score the shard, return
   the shard's top-k as global positions plus scores.
2. (backend) gather every rank's candidates.
3. ``dsa_topk_merge_dcp``: the global top-k, as the winners this rank owns.
4. (backend) gather the group's query heads.
5. ``sparse_mla_attention_fused_dcp``: store the owned latents, attend over the
   owned winners, return the partial output and its log-sum-exp.
6. (backend) merge the partials across the group by log-sum-exp.

The JAX bodies are in ``sglang_tpu.attention.pallas.dsa_index_topk`` and
``sglang_tpu.attention.pallas.mla_sparse_fused``. The rank enters every op as
an ``i32[1]`` tensor so all ranks run one compiled program.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

from typing import Optional

import torch

from sglang_tpu.pallas_export import (
    argnums,
    tpu_info_for_cpu_export,
)

_OPS: dict = {}


def build_dsa_index_topk_dcp_op(dcp_size: int, interleave_size: int):
    """Register ``dsa_indexer_step_dcp`` as a torch custom op, once per layout."""
    key = ("index", dcp_size, interleave_size)
    if key in _OPS:
        return _OPS[key]

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_index_topk import dsa_indexer_step_dcp

    op_name = f"sglang_tpu_pallas::dsa_index_topk_dcp{dcp_size}_i{interleave_size}"

    def _wrapped(
        query: jax.Array,
        key: jax.Array,
        head_weights: jax.Array,
        cache_k: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        dcp_rank: jax.Array,
        weight_scale: float,
        topk: int,
        segments: str,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return dsa_indexer_step_dcp(
            query,
            key,
            head_weights,
            cache_k,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            dcp_rank,
            weight_scale=weight_scale,
            topk=topk,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
            segments=segments,
        )

    (cache_argnum,) = argnums(_wrapped, "cache_k")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "weight_scale", "topk", "segments"),
        donate_argnums=(cache_argnum,),
        input_output_aliases={cache_argnum: 2},
        warn_deprecated=False,
    )

    @torch.library.custom_op(op_name, mutates_args=("cache_k",), device_types=["tpu"])
    def custom_op(
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        cache_k: torch.Tensor,
        seq_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        dcp_rank: torch.Tensor,
        weight_scale: float,
        topk: int,
        segments: str = "all",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with tpu_info_for_cpu_export():
            indices, scores, _updated_cache = jax_kernel(
                query,
                key,
                head_weights,
                cache_k,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                dcp_rank,
                weight_scale,
                topk,
                segments,
            )
        return indices, scores

    def _fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        cache_k: torch.Tensor,
        seq_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        dcp_rank: torch.Tensor,
        weight_scale: float,
        topk: int,
        segments: str = "all",
    ):
        shape = (query.shape[0], topk)
        return (
            torch.empty(shape, dtype=torch.int32, device=query.device),
            torch.empty(shape, dtype=torch.int32, device=query.device),
        )

    custom_op.register_fake(_fake_impl)
    _OPS[key] = custom_op
    return custom_op


def build_dsa_topk_merge_dcp_op(dcp_size: int, interleave_size: int):
    """Register ``merge_dcp_topk`` as a torch custom op, once per layout."""
    key = ("merge", dcp_size, interleave_size)
    if key in _OPS:
        return _OPS[key]

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_index_topk import merge_dcp_topk

    op_name = f"sglang_tpu_pallas::dsa_topk_merge_dcp{dcp_size}_i{interleave_size}"

    def _wrapped(
        cand_indices: jax.Array,
        cand_scores: jax.Array,
        dcp_rank: jax.Array,
        topk: int,
    ) -> jax.Array:
        return merge_dcp_topk(
            cand_indices,
            cand_scores,
            dcp_rank,
            topk=topk,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "topk"),
        warn_deprecated=False,
    )

    @torch.library.custom_op(op_name, mutates_args=(), device_types=["tpu"])
    def custom_op(
        cand_indices: torch.Tensor,
        cand_scores: torch.Tensor,
        dcp_rank: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            return jax_kernel(cand_indices, cand_scores, dcp_rank, topk)

    def _fake_impl(cand_indices, cand_scores, dcp_rank, topk):
        return torch.empty(
            (cand_indices.shape[1], topk),
            dtype=torch.int32,
            device=cand_indices.device,
        )

    custom_op.register_fake(_fake_impl)
    _OPS[key] = custom_op
    return custom_op


def build_sparse_mla_attention_dcp_op(dcp_size: int, interleave_size: int):
    """Register ``sparse_mla_attention_fused_dcp`` as a torch custom op."""
    key = ("attention", dcp_size, interleave_size)
    if key in _OPS:
        return _OPS[key]

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.mla_sparse_fused import (
        sparse_mla_attention_fused_dcp,
    )

    op_name = (
        f"sglang_tpu_pallas::sparse_mla_attention_fused_dcp{dcp_size}"
        f"_i{interleave_size}"
    )

    def _wrapped(
        ql_nope: jax.Array,
        q_pe: jax.Array,
        new_kv_c: jax.Array,
        new_k_pe: jax.Array,
        cache_nope: jax.Array,
        cache_rope: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        dcp_rank: jax.Array,
        topk_indices: jax.Array,
        sm_scale: float,
        k_scale: Optional[float],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return sparse_mla_attention_fused_dcp(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_nope,
            cache_rope,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            dcp_rank,
            topk_indices,
            sm_scale=sm_scale,
            k_scale=k_scale,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )

    nope_argnum, rope_argnum = argnums(_wrapped, "cache_nope", "cache_rope")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "sm_scale", "k_scale"),
        donate_argnums=(nope_argnum, rope_argnum),
        input_output_aliases={nope_argnum: 2, rope_argnum: 3},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name, mutates_args=("cache_nope", "cache_rope"), device_types=["tpu"]
    )
    def custom_op(
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        new_kv_c: torch.Tensor,
        new_k_pe: torch.Tensor,
        cache_nope: torch.Tensor,
        cache_rope: torch.Tensor,
        seq_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        dcp_rank: torch.Tensor,
        topk_indices: torch.Tensor,
        sm_scale: float,
        k_scale: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with tpu_info_for_cpu_export():
            output, lse, _updated_nope, _updated_rope = jax_kernel(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                cache_nope,
                cache_rope,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                dcp_rank,
                topk_indices,
                sm_scale,
                k_scale,
            )
        return output, lse

    def _fake_impl(ql_nope: torch.Tensor, *args, **kwargs):
        return (
            torch.empty_like(ql_nope),
            torch.empty(
                ql_nope.shape[:2], dtype=torch.float32, device=ql_nope.device
            ),
        )

    custom_op.register_fake(_fake_impl)
    _OPS[key] = custom_op
    return custom_op
