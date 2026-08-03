# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`objective.py`** — the judge: defines the goal, measures the run, computes the score. Not modified by the agent.
- **`tracking.py`** — MLflow logging for each experiment. Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py        — constants, data prep + runtime utilities (do not modify)
objective.py      — the goal, the meters, the score (do not modify)
tracking.py       — MLflow experiment logging (do not modify)
harness_check.py  — static guard on train.py's contract (do not modify)
train.py          — model, optimizer, training loop (agent modifies this)
program.md        — agent instructions
pyproject.toml    — dependencies
```

## The harness contract

`train.py` is rewritten every experiment, and most breakage is loud — the run
crashes and the ratchet discards it. Three failure modes are silent, because the
run still trains and still produces a score:

- a dropped `tracking.*` call, so the experiment never reaches MLflow
- `tracking.log_step` moved inside the timed window, so `training_seconds` and
  MFU quietly include network latency
- a training objective placed in `GPT.forward`, which is what `evaluate_bpb`
  calls — silently changing the number that decides whether that same experiment
  is kept

`harness_check.py` catches all three by parsing `train.py` — stdlib only, no
torch, no GPU, a few milliseconds:

```bash
uv run python harness_check.py
```

It runs two ways: CI fails on it, and `objective.py` calls it at import so a
broken edit shows up at the top of `run.log` inside the overnight loop, where CI
doesn't run. Checks are conservative — anything it can't establish confidently is
reported as `skipped` rather than failed, since a false alarm would just teach the
agent to work around the guard.

## Changing the goal

`objective.py` decides what "better" means. Its `GOAL` constant selects a scorer,
and the agent cannot reach the file:

```python
GOAL = "min_bpb"              # lowest bits/byte — the original goal
# GOAL = "min_bpb_under_vram" # same, but anything over VRAM_LIMIT_GB is rejected
# GOAL = "min_bpb_x_params"   # quality per parameter
```

The score is printed as `score:`, written to `run.json`, and appended to
`results.jsonl` along with every other metric. `alternative-goals.md` sketches the
goals this scaffolding was built for — latency-aware scoring, out-of-distribution
evaluation, token or FLOP budgets instead of wall clock.

Two structural rules make goal-swapping safe, and they're worth knowing before you
edit `train.py` by hand:

- **`GPT.forward` is the eval contract.** `evaluate_bpb` calls it and treats the
  result as plain cross-entropy. New training objectives go in `GPT.training_loss`,
  which is free to diverge — auxiliary heads, multi-token prediction, z-loss — with
  no effect on how the run is scored.
- **The judge owns the meters.** Parameter count, peak VRAM, FLOPs and wall clock
  are measured in `objective.py`, not reported by the file under test. That barely
  matters while the goal is `min_bpb`; it matters a great deal for any goal that
  scores a resource.

## Experiment tracking

Every run logs to MLflow automatically — hyperparameters, the per-step training
loss curve, the final metrics, and the exact `train.py` that produced them, so a
run in the UI stays self-describing after the branch has moved on. The default
server is `http://192.168.0.252:5000`; override it per-run or in your shell:

```bash
export MLFLOW_TRACKING_URI=http://your-host:5000   # where to log
export MLFLOW_EXPERIMENT_NAME=autoresearch         # experiment to log under
AUTORESEARCH_NOTE="increase LR to 0.04" uv run train.py
```

Tracking is best-effort by design. If the server is unreachable, `train.py` prints
one warning and trains normally — an overnight loop should never lose experiments
because a logging host went down. Two properties worth knowing:

- **Per-step metrics never touch the network.** They're buffered in memory and
  flushed once after training ends, because `train.py` times its own steps to
  enforce the 5-minute budget and report MFU. Logging inside that window would
  corrupt both numbers.
- **An unreachable server costs ~2s per run**, not the ~35s MLflow's default retry
  policy would spend, which matters when you're doing 100 runs a night.

To run a tracking server on your GPU box:

```bash
uv run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
