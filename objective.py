"""
objective.py — the judge.

Read-only, same status as prepare.py. train.py trains; this file decides what
"better" means and owns the measurements that decision rests on.

Two jobs:

  1. **Define the goal.** GOAL selects a scoring function. A scorer turns a
     finished run's metrics into a single number to minimize. Swapping research
     goals is a one-line edit here rather than a rewrite of train.py, and the
     agent cannot reach it.

  2. **Own the meters.** Parameter count, peak VRAM and wall clock are read here
     rather than trusted from the file under test. Under the default goal this is
     mostly hygiene — you cannot reach a lower val_bpb by mis-measuring your own
     memory. It stops being hygiene the moment a goal *scores* a resource, because
     then the meter is sitting inside the sandbox being optimized against.

Adding a goal: write a scorer, register it in SCORERS, point GOAL at it. Anything
needing a measurement train.py does not already produce — decode latency, an
out-of-distribution eval, a downstream task — belongs in this file too, for the
same reason.
"""

import os
import json
import time
import math
import subprocess

import torch

import harness_check
from prepare import evaluate_bpb

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPORT_TIME = time.time()

RESULTS_PATH = os.path.join(_REPO_DIR, "results.jsonl")
RUN_JSON_PATH = os.path.join(_REPO_DIR, "run.json")

# Warn — never fail — if train.py has drifted from its contract with the harness.
# CI is the hard gate, but CI does not run on autoresearch/* branches, so this is
# what puts a dropped tracking call at the top of run.log where the agent will
# see it. Costs a few milliseconds of AST parsing at startup.
harness_check.warn()

# ---------------------------------------------------------------------------
# The goal
# ---------------------------------------------------------------------------

GOAL = "min_bpb"  # <- the only line a human edits

VRAM_LIMIT_GB = 20.0  # used by min_bpb_under_vram


def _min_bpb(m):
    """Today's goal: lowest bits/byte in a fixed wall-clock budget."""
    return m["val_bpb"]


def _min_bpb_under_vram(m):
    """
    Constrained ratchet: same metric, but anything over the memory ceiling is
    rejected outright rather than traded off. Simpler to reason about than a
    scalarized penalty, and there are no weights to defend.
    """
    if m.get("peak_vram_gb") is None:
        return m["val_bpb"]
    return float("inf") if m["peak_vram_gb"] > VRAM_LIMIT_GB else m["val_bpb"]


def _min_bpb_x_params(m):
    """
    Quality per parameter. The exponent is small on purpose: bpb still dominates,
    and model size only breaks near-ties. Needs no measurement train.py doesn't
    already produce, which is why it's here and a latency goal isn't yet.
    """
    params_m = m.get("num_params_M") or 1.0
    return m["val_bpb"] * (params_m**0.05)


SCORERS = {
    "min_bpb": _min_bpb,
    "min_bpb_under_vram": _min_bpb_under_vram,
    "min_bpb_x_params": _min_bpb_x_params,
    # Goals from alternative-goals.md §4-5 plug in here. Each needs a measurement
    # this file would have to make for itself:
    #   min_bpb_x_latency  -> a decode benchmark (§5.1)
    #   min_bpb_ood        -> a second validation source (§5.2)
    #   time_to_target     -> periodic eval during training (§4.3)
}


def score(metrics):
    """Scalar to minimize. Lower is better, always, whatever GOAL is set to."""
    if GOAL not in SCORERS:
        raise ValueError(f"unknown GOAL {GOAL!r}; known: {sorted(SCORERS)}")
    return SCORERS[GOAL](metrics)


# ---------------------------------------------------------------------------
# Meters — measured here, not reported by train.py
# ---------------------------------------------------------------------------


def count_params(model):
    """Total parameters. Deliberately generic: sums whatever the model has."""
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    return sum(p.numel() for p in model.parameters())


_PEAK_FLOPS = None


def device_peak_flops(dtype=None):
    """
    Achievable dense bf16 throughput for THIS GPU, in FLOP/s — the denominator
    for MFU.

    Measured rather than looked up. A hardcoded H100 constant makes MFU
    meaningless on anything else (GB10 reports ~3% while doing perfectly
    reasonable work), and a table of vendor peak numbers goes stale every time
    new silicon appears. A big square matmul is a fair ceiling: it is what the
    hardware does when nothing is in its way.

    Override with AUTORESEARCH_PEAK_FLOPS if you want vendor peak instead.
    Cached — the measurement runs once per process, during startup, which is
    outside the training budget.
    """
    global _PEAK_FLOPS
    if _PEAK_FLOPS is not None:
        return _PEAK_FLOPS

    override = os.environ.get("AUTORESEARCH_PEAK_FLOPS")
    if override:
        _PEAK_FLOPS = float(override)
        return _PEAK_FLOPS

    fallback = 989.5e12  # H100 bf16, the historical constant
    if not torch.cuda.is_available():
        _PEAK_FLOPS = fallback
        return _PEAK_FLOPS
    try:
        dtype = dtype or torch.bfloat16  # resolved here, not as a default arg
        n, iters = 8192, 20
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(3):  # warm up
            a @ b
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        _PEAK_FLOPS = 2 * n**3 * iters / elapsed
        del a, b
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # don't charge the probe to the run
        print(f"Measured peak bf16 throughput: {_PEAK_FLOPS / 1e12:.1f} TFLOP/s")
    except Exception as e:
        print(
            f"[objective] peak FLOPS probe failed ({type(e).__name__}: {e}); "
            f"falling back to the H100 constant, so MFU will be wrong"
        )
        _PEAK_FLOPS = fallback
    return _PEAK_FLOPS


def peak_vram_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def wall_clock_seconds():
    """Seconds since this module was imported, i.e. roughly the whole process."""
    return time.time() - _IMPORT_TIME


def flops_per_token(model):
    """
    Analytic FLOPs/token estimate (forward + backward), moved out of train.py so
    a FLOP-budgeted goal would not be metering itself.

    Introspects a documented contract: model.config (n_head, n_embd,
    sequence_len), model.window_sizes, and which parameters are embeddings.
    Returns None rather than a wrong number if the model has been restructured
    past recognition — a missing MFU reading is honest, a fabricated one isn't.
    """
    model = getattr(model, "_orig_mod", model)
    try:
        config = model.config
        nparams = sum(p.numel() for p in model.parameters())
        # Embedding lookups and per-layer scalars are not matmuls.
        embedding_numel = sum(
            m.weight.numel()
            for m in model.modules()
            if isinstance(m, torch.nn.Embedding)
        )
        scalar_numel = sum(p.numel() for p in model.parameters() if p.ndim <= 1)
        dense = nparams - embedding_numel - scalar_numel

        h = config.n_head
        q = config.n_embd // config.n_head
        t = config.sequence_len
        attn_flops = 0
        for window_size in model.window_sizes:
            window = (
                window_size[0]
                if isinstance(window_size, (tuple, list))
                else window_size
            )
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * dense + attn_flops
    except Exception as e:
        print(f"[objective] could not estimate FLOPs ({type(e).__name__}: {e})")
        return None


def _git(*args, default=""):
    try:
        out = subprocess.run(
            ["git", "-C", _REPO_DIR, *args], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Reporting — one place, so the summary contract can't drift
# ---------------------------------------------------------------------------

# Printed in this order. Kept as `key: value` lines so the old greps still work.
_SUMMARY_ORDER = [
    "score",
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
]


def report(model, tokenizer, batch_size, stats):
    """
    Run the fixed evaluation, score it, and emit the record three ways: the
    human-readable summary block on stdout, run.json for machines, and an
    appended line in results.jsonl.

    `stats` is telemetry train.py alone can know (step count, training seconds,
    tokens seen). Everything measurable from outside is measured here instead.

    Call inside the same autocast context used for training.
    """
    val_bpb = evaluate_bpb(model, tokenizer, batch_size)

    num_params = count_params(model)
    metrics = {
        "val_bpb": val_bpb,
        **stats,
        # judge-measured, overriding anything of the same name in stats
        "num_params": num_params,
        "num_params_M": num_params / 1e6,
        "peak_vram_mb": peak_vram_mb(),
        "total_seconds": wall_clock_seconds(),
    }
    vram = metrics["peak_vram_mb"]
    metrics["peak_vram_gb"] = vram / 1024 if vram is not None else None

    # The sandbox reports its own training seconds (it has to — only it knows
    # which steps were compilation). Cross-check against wall clock so an
    # incoherent pair is visible in the record rather than silently scored.
    train_s, total_s = metrics.get("training_seconds"), metrics["total_seconds"]
    if train_s is not None and train_s > total_s + 1.0:
        metrics["timing_inconsistent"] = True
        print(
            f"[objective] WARNING: reported training_seconds={train_s:.1f} exceeds "
            f"measured wall clock {total_s:.1f}s"
        )

    metrics["score"] = score(metrics)
    metrics["goal"] = GOAL

    _print_summary(metrics)
    _write_run_json(metrics)
    _append_result(metrics)
    return metrics


_PRECISE = ("score", "val_bpb")  # the numbers decisions are made on


def _format(key, value):
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    return f"{value:.6f}" if key in _PRECISE else f"{value:.1f}"


def _print_summary(metrics):
    print("---")
    keys = [k for k in _SUMMARY_ORDER if k in metrics and metrics[k] is not None]
    keys += sorted(
        k
        for k in metrics
        if k not in _SUMMARY_ORDER and k != "goal" and metrics[k] is not None
    )
    print(f"{'goal:':20s}{metrics['goal']}")
    for key in keys:
        print(f"{key + ':':20s}{_format(key, metrics[key])}")


def _write_run_json(metrics):
    try:
        with open(RUN_JSON_PATH, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    except OSError as e:
        print(f"[objective] could not write run.json ({e})")


def _append_result(metrics):
    """
    Append this run to results.jsonl. Written by the harness rather than by the
    agent so a run cannot go unrecorded; the keep/discard verdict is filled in
    afterwards by record_decision().
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": _git("rev-parse", "--short=7", "HEAD", default="unknown"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", default="unknown"),
        "note": os.environ.get("AUTORESEARCH_NOTE", ""),
        "status": "pending",
        **{k: v for k, v in metrics.items()},
    }
    try:
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        print(f"[objective] could not append to results.jsonl ({e})")


def record_decision(status, note=None):
    """
    Stamp the most recent results.jsonl entry with keep / discard / crash.
    Called by the agent after it compares the score against the running best:

        uv run python -c "import objective; objective.record_decision('keep')"
    """
    if status not in ("keep", "discard", "crash"):
        raise ValueError(f"status must be keep/discard/crash, got {status!r}")
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"{RESULTS_PATH} does not exist yet")

    with open(RESULTS_PATH) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError("results.jsonl is empty")

    record = json.loads(lines[-1])
    record["status"] = status
    if note is not None:
        record["note"] = note
    lines[-1] = json.dumps(record, default=str)

    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        f"[objective] recorded {status} for commit {record.get('commit')} "
        f"(score={record.get('score')})"
    )


def log_crash(note=None):
    """
    Record a run that never reached report() — OOM, NaN, a broken edit. Keeps
    crashes in the same file as successes so the denominator stays honest.
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": _git("rev-parse", "--short=7", "HEAD", default="unknown"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", default="unknown"),
        "note": note if note is not None else os.environ.get("AUTORESEARCH_NOTE", ""),
        "status": "crash",
        "goal": GOAL,
        "score": None,
        "val_bpb": None,
    }
    # Never let a logging failure mask the crash we were called to record.
    try:
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[objective] recorded crash for commit {record['commit']}")
    except OSError as e:
        print(f"[objective] could not record crash ({e})")


def load_results(path=None):
    """Read results.jsonl into a list of dicts. Used by analysis.ipynb."""
    path = path or RESULTS_PATH  # resolved at call time, not import time
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def best_so_far(rows=None, goal=None):
    """Lowest score among kept runs, for the current goal."""
    rows = load_results() if rows is None else rows
    goal = goal or GOAL
    scored = [
        r
        for r in rows
        if r.get("status") == "keep"
        and r.get("goal") == goal
        and isinstance(r.get("score"), (int, float))
        and not math.isinf(r["score"])
    ]
    return min((r["score"] for r in scored), default=None)
