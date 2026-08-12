# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `objective.py` — the goal, the meters, the score. Do not modify. **Check which `GOAL` is set** — it determines what you are optimizing.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Note the goal**: Report back which `GOAL` is active and what the score therefore means. `results.jsonl` creates itself on the first run — nothing to initialize.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `objective.py`. It is read-only. It defines the goal, measures the run, and computes the score you are trying to lower.
- Modify `tracking.py`. It is read-only. It logs each experiment to MLflow.
- Change what `GPT.forward` returns. `prepare.evaluate_bpb` calls `model(x, y, reduction='none')` and treats the result as plain next-token cross-entropy. It is the scoreboard. Putting a different loss there silently changes the number that decides whether your own experiment gets kept. **New training objectives go in `GPT.training_loss` instead** — see below.
- Remove the harness calls in `train.py`: `objective.report`, `objective.log_crash`, and the four `tracking.*` calls. Rewrite everything around them freely, but carry them through. Keep `tracking.log_step` **outside** the `t0`/`t1` timing window; moving it inside would charge network latency to the time budget and corrupt the MFU number.

  This is checked. `uv run python harness_check.py` verifies it in a few milliseconds without touching the GPU, and every run prints `[harness] WARNING: ...` at the top of `run.log` if the contract is broken. **If you see one of those warnings, fix it before trusting the run** — the experiment will still train and still produce a score, which is exactly what makes this failure worth guarding. Run the check yourself after any large restructuring of `train.py`.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Set any environment variable other than `AUTORESEARCH_NOTE`, `AUTORESEARCH_TESTING_PLAN`, `AUTORESEARCH_AGENT`, and `AUTORESEARCH_MODEL`. The harness is configured through env vars (see `.env.example`), and several of them define the experiment rather than participate in it. `AUTORESEARCH_SEED` especially: changing seeds until one scores well is not a research result, it is shopping for a lucky initialisation, and the ratchet cannot tell the difference. `AUTORESEARCH_GOAL`, `AUTORESEARCH_TIME_BUDGET` and `AUTORESEARCH_EVAL_TOKENS` change what a score means, so a run under different values is not comparable to the ones before it.

**The goal: get the lowest `score`.** The script prints it; `objective.py` defines it. Under the default goal (`min_bpb`) the score is just `val_bpb`, so this is the same thing as before — but read the `goal:` line rather than assuming, because the human can change it between runs and a different goal may price in memory, model size or other costs.

Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. The only constraint is that the code runs without crashing and finishes within the budget.

**Two places to change the model, and they mean different things:**
- `GPT.forward` — the eval contract. Architecture changes belong here (attention, MLP, normalization, embeddings, depth, width). What must not change is that with `targets` supplied it returns plain cross-entropy.
- `GPT.training_loss` — the training objective, and it is allowed to diverge from `forward`. Auxiliary heads, multi-token prediction, z-loss, label smoothing, distillation, per-position loss weighting: all fair game here, and none of it touches evaluation. `GPT.trunk` gives you the hidden states if you want to hang extra heads off them.

**VRAM** is a soft constraint under the default goal. Some increase is acceptable for meaningful gains, but it should not blow up dramatically. (If the goal is set to `min_bpb_under_vram` it stops being soft — over the limit scores as infinity and is always a discard.)

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 score improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 score improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
goal:               min_bpb
score:              0.997900
val_bpb:            0.997900
training_seconds:   300.1
total_seconds:      325.9
peak_vram_mb:       45060.2
mfu_percent:        39.8
total_tokens_M:     499.6
num_steps:          953
num_params_M:       50.3
depth:              8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. Extract the key metrics from the log file:

```
grep "^goal:\|^score:\|^val_bpb:\|^peak_vram_mb:" run.log
```

The same numbers are written to a `run-<timestamp>.json` in machine-readable form (a fresh file per run, so it never overwrites the previous one), which is easier if you want more than one metric:

```
cat "$(ls -t run-*.json | head -1)"
```

## Logging results

Every run also logs itself to MLflow automatically (hyperparameters, the training
loss curve, the final metrics, and a copy of the `train.py` that produced them).
That happens without you doing anything beyond setting `AUTORESEARCH_NOTE`. If the
tracking server is unreachable the run prints a one-line warning and continues
normally — that is not a failure, do not try to fix it, and do not let it change
your keep/discard decision.

`results.jsonl` is the ratchet's own record, and **you do not write it by hand**.
Every run appends its own line automatically — commit, branch, note, goal, score
and every metric — with `"status": "pending"`. Crashes caught by the divergence
guard append themselves too. This means a run can never go unrecorded because you
forgot.

The one thing left to you is the verdict. After you decide keep or discard:

```
uv run python -c "import objective; objective.record_decision('keep')"
```

That stamps the most recent entry. Valid values are `keep`, `discard`, `crash`.
If a run died in a way that left no entry at all (a hard crash before training
started), record it explicitly:

```
uv run python -c "import objective; objective.log_crash('OOM at 2x width')"
```

Each line looks roughly like this — the exact keys vary with the goal, which is
the point of using JSON rather than fixed columns:

```json
{"timestamp": "2026-08-04T02:14:07", "commit": "a1b2c3d", "branch": "autoresearch/aug4",
 "note": "increase LR to 0.04", "testing_plan": "experiment-catalogue.md", "agent": "hermes",
 "model": "qwen3.6:35b", "status": "keep", "goal": "min_bpb", "score": 0.9932, "val_bpb": 0.9932,
 "peak_vram_gb": 44.2, "num_params_M": 50.3, "num_steps": 953}
```

Do not commit `results.jsonl` or `run-*.json` — both are gitignored.

### Snapshotting results for shared history

`results.jsonl` is per-machine and gitignored, so it never survives a fresh
clone and isn't visible across different agents/sandboxes working this repo.
Periodically — roughly every 20-30 experiments, or before ending a session —
snapshot it onto `main` so the history survives:

```
git checkout main
cp results.jsonl results/results-$(date +%m%d%Y).jsonl
git add results/results-$(date +%m%d%Y).jsonl
git commit -m "Snapshot results.jsonl through $(date +%Y-%m-%d)"
git push
git checkout <your working branch>
```

Do this from `main`, never from your working ratchet branch — a working
branch gets `git reset` on every discard, and a snapshot committed there
could be lost along with everything else in that reset. Files under
`results/` are additive-only: never overwrite an existing dated snapshot,
just add a new one covering whatever's been appended since.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment, passing a one-line description, the testing plan you're following, and which agent/model is driving the run so the record is labelled:
   `AUTORESEARCH_NOTE="increase LR to 0.04" AUTORESEARCH_TESTING_PLAN="experiment-catalogue.md" AUTORESEARCH_AGENT="hermes" AUTORESEARCH_MODEL="qwen3.6:35b" uv run train.py > run.log 2>&1`
   (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^goal:\|^score:\|^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up — and if nothing was appended to `results.jsonl`, call `objective.log_crash(...)` so the failure is still counted.
7. Record your verdict: `uv run python -c "import objective; objective.record_decision('keep')"` (or `'discard'`)
8. If the score improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ startup, compile and eval overhead). Rather than watching the clock, put the limit in the command so a pathological config cannot eat the night:

```
AUTORESEARCH_NOTE="..." timeout 900 uv run train.py > run.log 2>&1
```

A run killed by `timeout` exits 124 and writes no summary. Treat it as a failure: record it with `objective.log_crash('timed out')` and revert. This matters more than it looks — the time budget only starts counting after 10 warmup steps, so a config whose individual steps are enormously slow can run far past 5 minutes without ever tripping the budget.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
