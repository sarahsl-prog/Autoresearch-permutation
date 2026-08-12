# AutoResearch Experiment Plan – aug10

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task.

## Goal
Launch the first autonomous experiment on the current branch (`autoresearch/aug10`) by:
1. Ensuring data & tokenizer are prepared.
2. Running a baseline training script for the fixed 5‑minute time budget.
3. Recording the run’s metrics and deciding to **keep** or **discard**.
4. Stamping the decision into `results.jsonl`.

## Current Context
- Working directory: `/sandbox/Autoresearch-permutation`
- Active git branch: `autoresearch/aug10` (new branch created from `autoresearch/baseline-20260810`)
- Python toolchain managed by `uv`; dependencies installed (`uv sync` succeeded)
- Cache directories:
  - `/sandbox/.cache/autoresearch/data/` – contains downloaded parquet shards
  - `/sandbox/.cache/autoresearch/tokenizer/` – trained BPE tokenizer
- `program.md` defines that the experiment loop runs **forever** until manually stopped.
- The default goal in `objective.py` is `min_bpb` (bits‑per‑byte) which we must minimize.

## Proposed Approach
1. **Verify data availability.**  
   - Check that at least one training shard and the validation shard exist under `~/.cache/autoresearch`.  
   - If missing, run a minimal download with `prepare.py --num-shards 2`.

2. **Set experiment‑level configuration.**  
   - Export `AUTORESEARCH_NOTE="baseline‑aug10"` to label the run.  
   - Optionally set `AUTORESEARCH_SEED` for repeatability (default is 42).

3. **Run a baseline training job** with a strict 5‑minute wall‑clock limit:
   ```bash
   timeout 900 uv run train.py > run.log 2>&1
   ```
   - This respects the time budget enforced by `prepare.py` (`TIME_BUDGET=300`).

4. **Extract key metrics** from `run.log` and a generated JSON snapshot:
   ```bash
   grep "^goal:\|^score:\|^val_bpb:\|^peak_vram_mb:" run.log
   cat "$(ls -t run-*.json | head -1)"
   ```

5. **Decide keep vs. discard.**  
   - Compare the obtained `score` against the best score currently recorded in `results.jsonl`.  
   - If the new score is strictly lower, record a *keep*; otherwise *discard*.

6. **Stamp the decision** using the harness helper:
   ```bash
   uv run python -c "import objective; objective.record_decision('keep')"
   ```
   or `...'discard'` as appropriate.

7. **(Optional) Advance the branch.**  
   - If kept, simply stay on `autoresearch/aug10`.  
   - If discarded, reset to the pre‑run commit (`git reset --hard HEAD~1`) and repeat step 3 with a new edit.

## Files Likely to Be Touched
| Action | Path (relative) |
|--------|-----------------|
| Create experiment plan file (this one) | `.hermes/plans/aug10-experiment-plan.md` |
| Export run note environment variable | (shell command – no file) |
| Run training script | `train.py` |
| Capture log & metrics | `run.log`, generated `run-*.json` |
| Record decision | `results.jsonl` (appended by harness) |

## Verification Steps
1. **Data sanity check** – after running `prepare.py --num-shards 2` you should see:
   ```
   Cache directory: /sandbox/.cache/autoresearch
   Data: all 3 shards already downloaded at /sandbox/.cache/autoresearch/data
   Tokenizer: already trained …
   Done! Ready to train.
   ```
2. **Training termination** – `timeout 900` must exit with code 124 after ~900 seconds; `run.log` should contain a summary block printed by `objective.report`. Example grep command above must return lines starting with `goal:`, `score:` etc.
3. **Decision recording** – `results.jsonl` must gain a new line whose `"status"` field is either `keep` or `discard`.

## Risks & Trade‑offs
- **Network availability**: If the HuggingFace download fails, subsequent runs will error out; mitigate by pre‑seeding shards manually.
- **GPU memory contention**: Overly large model edits can push VRAM > default limit (20 GB). In that case either lower `DEPTH` or switch the goal to `min_bpb_under_vram`.
- **Flaky timing**: The 5‑minute wall clock includes startup/compilation; unusually slow steps may cause a timeout even though the actual training loop finishes earlier. Use `timeout 900` (15 min) as a safe margin.

## Next Action
Execute step 3 of the plan: launch the baseline training run with the configured time‑budget and log capture.