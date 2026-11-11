"""Raiden pull transfer into a sparse MLA decode-context-parallel decode.

Every prefill page must land exactly once, on DCP rank ``s % dcp`` at that
rank's ``s // dcp``-th page, which is where the sparse MLA DCP kernels read it.
"""

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.receiver import RaidenReceiverMixin
from sglang_tpu.disaggregation.raiden.sender import _dcp_stripe
from sglang_tpu.disaggregation.raiden.serde import PullReq

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("num_pages", [1, 3, 8, 13, 64, 65])
@pytest.mark.parametrize("reserved", [0, 1])
def test_every_prefill_page_lands_on_its_owner(dcp_size, num_pages, reserved):
    prefill_pages = [1000 + s for s in range(num_pages)]
    # Decode pages are widened (page_size * dcp tokens); one extra may be reserved
    # for the first decode token.
    decode_pages = -(-num_pages // dcp_size) + reserved
    placed = {}
    for rank in range(dcp_size):
        local = [10 * rank + k for k in range(decode_pages)]
        remote, paired = RaidenReceiverMixin._pair_blocks(
            _dcp_stripe(prefill_pages, dcp_size=dcp_size, dcp_rank=rank), local
        )
        assert len(remote) == len(paired)
        for src, dst in zip(remote, paired):
            assert src not in placed
            placed[src] = (rank, dst)
    assert placed == {
        1000 + s: (s % dcp_size, 10 * (s % dcp_size) + s // dcp_size)
        for s in range(num_pages)
    }


def test_pull_req_carries_dcp():
    req = PullReq(
        room=7, decode_ip="10.0.0.1", decode_port=4000, decode_pages=3,
        dcp_size=8, dcp_rank=5,
    )
    back = PullReq.from_frames(req.to_frames())
    assert (back.dcp_size, back.dcp_rank) == (8, 5)
    plain = PullReq(room=7, decode_ip="10.0.0.1", decode_port=4000, decode_pages=3)
    assert (plain.dcp_size, plain.dcp_rank) == (1, 0)


def test_pull_req_rejects_out_of_range_rank():
    with pytest.raises(ValueError):
        PullReq(
            room=7, decode_ip="10.0.0.1", decode_port=4000, decode_pages=3,
            dcp_size=4, dcp_rank=4,
        )
