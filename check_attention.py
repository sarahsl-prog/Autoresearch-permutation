"""
check_attention.py — validate the attention backend before spending a training run.

train.py's FA3 path fails off Hopper: the kernels-community build is not an
opaque custom op, so torch.compile traces into it and dies during fake-tensor
propagation. The SDPA backend replaces it. This script checks that replacement
numerically and, more importantly, checks that it *compiles* — which is exactly
what broke.

    uv run python check_attention.py

Deliberately standalone. train.py has no __main__ guard, so importing it would
start a 5-minute training run; the ~15 lines under test are duplicated here
rather than imported. Keep them in sync if you change train.py's attention.
"""

import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)

B, H, T, D = 2, 4, 256, 128
WINDOW = 64
DEV = "cuda"


def reference(q, k, v, window=None):
    """Naive fp32 attention. Slow and obviously correct — the thing to trust."""
    q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))  # (B, H, T, D)
    scores = (q @ k.transpose(-2, -1)) / (D**0.5)
    i = torch.arange(T, device=q.device)
    allowed = i[:, None] >= i[None, :]
    if window is not None:
        allowed = allowed & ((i[:, None] - i[None, :]) <= window)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(1, 2)  # (B, T, H, D)


def sdpa_attention(q, k, v, block_mask, n_head, n_kv_head):
    """Copy of train.py's helper."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if n_kv_head != n_head:
        repeat = n_head // n_kv_head
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    if block_mask is None:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        y = flex_attention(q, k, v, block_mask=block_mask)
    return y.transpose(1, 2)


def report(name, got, want, tol=2e-2):
    err = (got.float() - want.float()).abs().max().item()
    ok = err < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} max abs err {err:.5f}")
    return ok


if not torch.cuda.is_available():
    raise SystemExit("no CUDA device")

cap = torch.cuda.get_device_capability()
print(f"device      : {torch.cuda.get_device_name(0)} (capability {cap[0]}.{cap[1]})")
print(f"torch       : {torch.__version__} (cuda {torch.version.cuda})")
print(f"arch list   : {torch.cuda.get_arch_list()}")
print(f"shapes      : B={B} H={H} T={T} D={D} window={WINDOW}\n")

from torch.nn.attention.flex_attention import flex_attention, create_block_mask  # noqa: E402

q, k, v = (torch.randn(B, T, H, D, device=DEV, dtype=torch.bfloat16) for _ in range(3))

print("== correctness (eager) ==")
ok = True
ok &= report(
    "full causal (SDPA is_causal)",
    sdpa_attention(q, k, v, None, H, H),
    reference(q, k, v),
)


def sliding_causal(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= WINDOW)


t0 = time.time()
block_mask = create_block_mask(
    sliding_causal, B=None, H=None, Q_LEN=T, KV_LEN=T, device=DEV
)
print(f"  (create_block_mask took {time.time() - t0:.1f}s)")
ok &= report(
    "sliding window (FlexAttention)",
    sdpa_attention(q, k, v, block_mask, H, H),
    reference(q, k, v, WINDOW),
)

print("\n== GQA (n_kv_head < n_head) ==")
kv = (torch.randn(B, T, H // 2, D, device=DEV, dtype=torch.bfloat16) for _ in range(2))
k2, v2 = kv
k2e = k2.repeat_interleave(2, dim=2)
v2e = v2.repeat_interleave(2, dim=2)
ok &= report(
    "expanded kv heads",
    sdpa_attention(q, k2, v2, None, H, H // 2),
    reference(q, k2e, v2e),
)

# The real test: this is where FA3 died.
print("\n== compiles (this is what FA3 failed) ==")
compiled = torch.compile(sdpa_attention, dynamic=False)
try:
    t0 = time.time()
    y = compiled(q, k, v, None, H, H)
    torch.cuda.synchronize()
    print(
        f"  PASS  full causal compiled          ({time.time() - t0:.1f}s incl. compile)"
    )
    ok &= report("compiled output matches eager", y, reference(q, k, v))
except Exception as e:
    ok = False
    print(f"  FAIL  full causal compiled: {type(e).__name__}: {e}")

try:
    t0 = time.time()
    y = compiled(q, k, v, block_mask, H, H)
    torch.cuda.synchronize()
    print(
        f"  PASS  sliding window compiled       ({time.time() - t0:.1f}s incl. compile)"
    )
    ok &= report("compiled output matches eager", y, reference(q, k, v, WINDOW))
except Exception as e:
    ok = False
    print(f"  FAIL  sliding window compiled: {type(e).__name__}: {e}")

# train.py's "S" layers use window = sequence_len // 2, not the small WINDOW used
# for the correctness checks above. Timing with a 64-token window would flatter
# the banded path enormously and tell you nothing about the real run.
TRAIN_T, TRAIN_WINDOW, TRAIN_H, TRAIN_D = 2048, 1024, 4, 128
BENCH_B = 8  # train.py uses DEVICE_BATCH_SIZE=128; scale accordingly

print(f"\n== throughput (T={TRAIN_T}, window={TRAIN_WINDOW}, B={BENCH_B}) ==")


def train_sliding(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= TRAIN_WINDOW)


try:
    qb, kb, vb = (
        torch.randn(
            BENCH_B, TRAIN_T, TRAIN_H, TRAIN_D, device=DEV, dtype=torch.bfloat16
        )
        for _ in range(3)
    )
    big_mask = create_block_mask(
        train_sliding, B=None, H=None, Q_LEN=TRAIN_T, KV_LEN=TRAIN_T, device=DEV
    )
    for label, m in (("full causal (L)", None), ("sliding window (S)", big_mask)):
        compiled(qb, kb, vb, m, TRAIN_H, TRAIN_H)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            compiled(qb, kb, vb, m, TRAIN_H, TRAIN_H)
        torch.cuda.synchronize()
        per_iter = (time.time() - t0) / 10 * 1000
        print(f"  {label:20s} {per_iter:6.2f} ms/iter")
    print(f"  (one attention call; train.py runs {8} layers per fwd pass)")
except Exception as e:
    print(f"  skipped ({type(e).__name__}: {e})")

print(
    f"\n{'OK: attention backend is usable' if ok else 'FAILURES — do not start a run'}"
)
raise SystemExit(0 if ok else 1)
