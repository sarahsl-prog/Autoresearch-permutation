# Experiment catalogue - Claude Opus

Fifty-three experiments across seven categories, sized for this machine and this
baseline. Written to be run by hand, but every entry is also a fair prompt for the
autonomous loop.

| Category | Tests | What it's for |
|---|---|---|
| [A — Calibration](#category-a--calibration) | 6 | What a real result looks like. **Do this first.** |
| [B — Model shape](#category-b--model-shape) | 7 | Depth, width, and the embedding imbalance |
| [C — Optimization](#category-c--optimization) | 10 | Hyperparameters that may not have survived the move off H100 |
| [D — Training objective](#category-d--training-objective) | 8 | What the model is asked to learn |
| [E — Architecture](#category-e--architecture) | 8 | Attention, MLP, normalization, embeddings |
| [F — Goal & harness](#category-f--goal-and-harness-meta-experiments) | 6 | Changing what "better" means |
| [G — Deliberately interesting](#category-g--deliberately-interesting) | 8 | Won't win; will teach |

★ marks the entries with the best information-per-run.

---

## The baseline everything is measured against

From the run at commit `0052cdd`:

| | |
|---|---|
| `val_bpb` | **1.335618** |
| `num_steps` | 582 |
| tokens seen | 152.6M |
| throughput | 497,934 tok/s |
| `mfu_percent` | 28.5% (against a measured 99.0 TFLOP/s) |
| peak VRAM | 15.5 GB of 128 GB |
| params | 11.5M — of which **8.4M are embeddings** |
| wall clock | 348s (300s training + ~46s eval + startup) |

Config: `DEPTH=4`, `n_embd=256`, `n_head=2`, `HEAD_DIM=128`, `WINDOW_PATTERN="L"`,
`TOTAL_BATCH_SIZE=2**18`, `DEVICE_BATCH_SIZE=128`, `grad_accum_steps=1`.

**Cost model: 5.8 min/run → ~10 runs/hour → ~83 runs in an eight-hour night.**
A 5-run sweep costs 29 minutes. Budget accordingly; the catalogue below is far
more than one night.

---

## Protocol

Rules that make the difference between experiments and anecdotes.

1. **Do category A first.** Until you know the noise floor, no other result here
   can be interpreted. A 0.003 "improvement" is meaningless if seed variance is
   0.01.
2. **One change per run.** Two changes that each help slightly and together hurt
   are indistinguishable from noise you can't attribute.
3. **Label every run**: `AUTORESEARCH_NOTE="A1 seed 43"`. The ID goes in
   `results.jsonl` and MLflow, and the analysis is unreadable without it.
4. **Never touch `AUTORESEARCH_SEED` to chase a score.** Vary it only to *measure*
   variance. Picking the seed that scored best is not a result.
5. **Cap the run**: `timeout 900 uv run train.py > run.log 2>&1`. The time budget
   only starts counting after 10 warmup steps, so a config with very slow steps
   can run far past five minutes without tripping it.
6. **Record crashes too.** `objective.log_crash('...')` — a category where half
   the ideas OOM is itself a finding, and it keeps the denominator honest.
7. **Watch `num_steps` as well as `val_bpb`.** Anything that drops below ~300
   steps re-breaks the Muon momentum ramp, and you'll be measuring the schedule
   mismatch rather than the idea.

### Where things live

| What | Where |
|---|---|
| Hyperparameters | `train.py`, the block near "Hyperparameters (edit these directly)" |
| Model architecture | `train.py` — `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` |
| **Training objective** | `train.py` — `GPT.training_loss` (free to diverge from `forward`) |
| Eval contract | `train.py` — `GPT.forward` **must keep returning plain cross-entropy** |
| Goal / meters / seed | `objective.py` (read-only to the agent; env-configurable) |
| Harness constants | `prepare.py` (read-only; `TIME_BUDGET` and `EVAL_TOKENS` via env) |

Two constraints that will bite:

- `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * 2048) == 0` — asserted at startup.
- `model_dim = ceil(DEPTH * ASPECT_RATIO / HEAD_DIM) * HEAD_DIM` and
  `n_head = model_dim / HEAD_DIM`. `DEPTH=4, ASPECT_RATIO=64, HEAD_DIM=128` gives
  `model_dim=256, n_head=2`. Small changes to `DEPTH` can silently change head
  count.

---

## Category A — Calibration

**Run this category before anything else.** These don't improve the model; they
tell you which later results are real. Roughly 20 runs, two hours.

### A1. Seed noise floor ★ start here
Same commit, `AUTORESEARCH_SEED` ∈ {41,42,43,44,45}. **5 runs.**
Report mean and standard deviation of `val_bpb`.
→ **This number becomes your keep-threshold.** Improvements smaller than ~1σ are
noise. Expect σ somewhere in 0.002–0.015; if it's at the high end, most of what
an overnight ratchet "discovers" will be luck.

### A2. Eval-sampling noise
Fixed seed, `AUTORESEARCH_EVAL_TOKENS` ∈ {2621440, 5242880, 10485760, 20971520}. **4 runs.**
→ Separates *eval* noise from *training* noise. If val_bpb barely moves as eval
shrinks 8×, you can cut eval to ~5M tokens and buy back ~30s per run — roughly
**10 extra experiments per night**. Interesting either way.

### A3. Determinism
Same seed, same commit, twice. **2 runs.**
→ Is the pipeline bitwise reproducible? `torch.compile` + SDPA + non-deterministic
reductions often aren't. If two identical runs differ, that difference is a *floor*
under A1 and everything else.

### A4. Budget transfer ★ high value
Baseline and one variant known to differ, each at `AUTORESEARCH_TIME_BUDGET` 300
and 900. **4 runs, ~35 min.**
→ Does a 5-minute ranking survive at 15 minutes? If not, the entire overnight
premise is measuring something that doesn't transfer. This is the single most
important negative result available here.

### A5. Train/val gap
Log final training loss alongside `val_bpb` and compare across A1's five runs.
**0 extra runs** — read the MLflow curves.
→ At 582 steps and 152M tokens on a pinned val shard, is there any overfitting
signal at all? If not, regularisation experiments (C6, D3) are probably dead ends.

### A6. Val-shard sensitivity
Point eval at a different shard (requires a small local edit to `VAL_SHARD`; keep
it out of your ratchet branch). **3 runs.**
→ How much of `val_bpb` is a property of *that one shard*? Selection pressure over
~100 experiments against a single fixed sample is the classic overfitting-to-val
setup, and this bounds it.

---

## Category B — Model shape

The parameter budget is oddly distributed: 8.4M of 11.5M is embeddings, and only
3.1M is actual transformer. Against **non-embedding** params you're at ~49
tokens/param (overtrained); against total params, ~13 (undertrained). Several
tests here poke directly at that.

### B1. Depth sweep
`DEPTH` ∈ {2, 3, 4, 6, 8}. **5 runs.** Note `DEPTH=2` → `model_dim=128, n_head=1`;
`DEPTH=6` → `model_dim=384, n_head=3`.
→ Is 4 actually the optimum at this budget, or just where we landed? Plot
`val_bpb` against `num_steps` as well — depth trades steps for capacity.

### B2. Aspect ratio
`ASPECT_RATIO` ∈ {32, 64, 96, 128} at `DEPTH=4`. **4 runs.**
→ Width at fixed depth. `ASPECT_RATIO=128` gives `model_dim=512, n_head=4`,
quadrupling transformer params without touching embeddings — a direct probe of
the imbalance above.

### B3. Head dimension
`HEAD_DIM` ∈ {64, 128} at fixed `model_dim=256`. **2 runs.**
→ 4 small heads vs 2 large. Cheap, and head count interacts with everything in
Category E.

### B4. Tied embeddings ★ interesting
Share `lm_head.weight` with `transformer.wte.weight`. Frees 2.1M params (18% of
the model) with no throughput cost. **1 run**, plus **1** reinvesting the savings
in width.
→ Classic small-model win. If tying helps *and* the reinvested version helps more,
the imbalance hypothesis is confirmed.

### B5. Drop value embeddings ★ simplification
Make `has_ve` return `False` everywhere. Removes 4.2M params — 36% of the model.
**1 run**, plus **1** reinvesting into `ASPECT_RATIO`.
→ ResFormer value embeddings were tuned at 50M+ params on an H100. At 11.5M they
may be a poor use of a third of the budget. Equal-or-better here is a large
simplification win under `program.md`'s criterion.

### B6. Value-embedding rank
Keep value embeddings but factor them: `vocab → r → kv_dim` with r ∈ {32, 64}.
**2 runs.**
→ Middle ground between B5 and baseline. Interesting even if it loses, because it
localises *where* the value-embedding benefit lives.

### B7. Matched-parameter depth/width
Pick three (`DEPTH`, `ASPECT_RATIO`) pairs with equal total params. **3 runs.**
→ Isolates *shape* from *size*. The cleanest scaling result in this catalogue.

---

## Category C — Optimization

Learning rates were tuned upstream at 524K-token batches on an H100; you're at
262K on different silicon. Some of this is almost certainly mistuned.

### C1. Warmup ★ likely mistuned
`WARMUP_RATIO` ∈ {0.0, 0.01, 0.02, 0.05, 0.10}. **5 runs.**
→ Currently **zero**: full LR from step 1. Defensible at 90 steps, questionable at
582 with a halved batch. My first guess at a real win.

### C2. Warmdown shape
`WARMDOWN_RATIO` ∈ {0.2, 0.35, 0.5, 0.7}. **4 runs.**
→ At 0.5 the LR starts decaying at step 291. With more steps now available, a
later, sharper decay may beat it.

### C3. Final LR floor
`FINAL_LR_FRAC` ∈ {0.0, 0.02, 0.1}. **3 runs.**
→ Does annealing all the way to zero waste the last steps?

### C4. Matrix LR
`MATRIX_LR` ∈ {0.02, 0.03, 0.04, 0.06, 0.08}. **5 runs.**
→ The Muon LR. Halving the batch usually wants a lower LR; the `1/sqrt(dmodel)`
scaling already *raised* it (256 vs 768 → ×1.73). Those pull opposite ways, so
this is genuinely unknown.

### C5. Embedding / unembedding LR
`EMBEDDING_LR` ∈ {0.3, 0.6, 1.0}, then `UNEMBEDDING_LR` ∈ {0.002, 0.004, 0.008}. **6 runs.**
→ With embeddings at 73% of params, these matter more here than upstream.

### C6. Weight decay
`WEIGHT_DECAY` ∈ {0.0, 0.1, 0.2, 0.4}. **4 runs.**
→ Pair with A5. If there's no overfitting signal, expect 0.0 to win, and that's a
simplification.

### C7. Batch size ★
`TOTAL_BATCH_SIZE` ∈ {2**16, 2**17, 2**18, 2**19} (adjust `DEVICE_BATCH_SIZE` to
keep the assert satisfied). **4 runs.**
→ The steps-vs-gradient-quality trade, now measurable end to end. Watch
`num_steps` in each.

### C8. Muon momentum ramp
`get_muon_momentum`'s 300-step ramp, ∈ {100, 300, 600}. **3 runs.**
→ At 90 steps this never completed; at 582 it's meaningful for the first time.
Worth knowing whether it was ever load-bearing.

### C9. Newton–Schulz steps
`ns_steps` ∈ {3, 5, 7} in `setup_optimizer`. **3 runs.**
→ Orthogonalisation quality against per-step cost. Fewer steps = more steps
overall. A genuine throughput/quality trade.

### C10. Adam betas
`ADAM_BETAS` ∈ {(0.8,0.95), (0.9,0.95), (0.9,0.99)}. **3 runs.**
→ Cheap, and short runs often prefer lower β₁.

---

## Category D — Training objective

These need `GPT.training_loss`, which is free to diverge from `forward`. Evaluation
is unchanged, so results stay directly comparable to the baseline — the reason the
split exists.

### D1. Multi-token prediction ★
Auxiliary heads predicting t+2…t+k off `GPT.trunk`, weighted `AUX_WEIGHT`, ignored
at eval. Sweep k ∈ {2,3}, `AUX_WEIGHT` ∈ {0.1, 0.3}. **4 runs.**
→ Well-precedented (Gloeckle et al., DeepSeek-V3). The heads cost throughput, so
it's a real trade rather than a free lunch.

### D2. z-loss ★ possible simplification
Add `λ·log²Z` to `training_loss`, λ ∈ {1e-4, 1e-3}. **2 runs**, plus **1** with
the logit softcap removed entirely.
→ z-loss and the `softcap=15` tanh do overlapping jobs. If z-loss lets you delete
the softcap at equal loss, that's a simplification win *and* removes a term from
the eval path.

### D3. Label smoothing
ε ∈ {0.01, 0.05}. **2 runs.**
→ Usually hurts bpb, occasionally helps under short budgets. Pair with A5.

### D4. Position-weighted loss ★ interesting
The dataloader packs documents BOS-aligned, so early positions in each row are
systematically easier. Down-weight the first N positions, N ∈ {8, 32}. **2 runs.**
→ Three lines. Directly targets a known property of *this* data pipeline rather
than a generic trick, which makes it more likely to be real.

### D5. Sequence-length curriculum
Reshape loader output `(B,2048) → (4B,512)` early, anneal to full length. **2 runs.**
→ Cheap attention early buys steps. Works without touching `prepare.py`: the
loader hands you tensors, and `train.py` may rearrange them however it likes.

### D6. EMA self-distillation
Slow EMA copy of the model as teacher, consistency loss against its soft targets.
**2 runs.**
→ Plausible under short budgets where the model is far from converged. May just
be an expensive LR schedule — which is itself worth knowing.

### D7. Token-frequency reweighting
Down-weight the loss on the most frequent tokens. **2 runs.**
→ bpb is byte-weighted, so token-frequency effects don't map cleanly onto the
metric. Interesting precisely because the mismatch is hard to predict.

### D8. Fill-in-the-middle
Permute spans within `x`, using the 3 unused reserved tokens as sentinels. **1 run.**
→ **Will hurt `val_bpb` by construction.** Run it anyway, once, to see the size of
the penalty — it calibrates how much a capability costs under a pure-likelihood
goal, and motivates Category F.

---

## Category E — Architecture

### E1. GQA
`n_kv_head` ∈ {1, 2} at `n_head=2`, and revisit at `n_head=4` (B2). **2–3 runs.**
→ `n_kv_head == n_head` today. The SDPA path already expands kv heads. Frees
parameters and shrinks the KV cache — the latter matters only if you later adopt
an inference-time goal (F4).

### E2. Window pattern, now that both paths work
`WINDOW_PATTERN` ∈ {"L", "SSSL", "SL", "SSSS"}. **4 runs.**
→ Baseline is all-`L` because I forced it for stability. `"S"` layers route through
FlexAttention; `"L"` through `is_causal`. Compare `val_bpb` **and** `num_steps` —
the two backends have different cost profiles and the comparison is not upstream's.

### E3. Activation
`relu()²` → `gelu`, `silu`, SwiGLU. **3 runs.**
→ SwiGLU changes parameter count; hold total params fixed by trimming the MLP
ratio, or the comparison is really E4.

### E4. MLP ratio
`4 * n_embd` → {2, 3, 6}×. **3 runs.**
→ Moves budget between attention and MLP. With attention already cheap at
`n_head=2`, the optimum may not be 4.

### E5. QK-norm ablation
Remove `norm(q), norm(k)`. **1 run.**
→ A stability measure that may cost quality. If removing it is neutral, that's a
simplification.

### E6. Rotary base
`base` ∈ {1000, 10000, 50000} in `_precompute_rotary_embeddings`. **3 runs.**
→ 10000 is a convention inherited from much longer contexts. At 2048 with a small
model, unexamined.

### E7. Softcap
`softcap` ∈ {10, 15, 30, off}. **4 runs.**
→ Note this term sits in `logits_from`, on the **eval path** — changing it changes
what `evaluate_bpb` measures. Legitimate, but flag it in the note, and read D2 first.

### E8. Residual scalar init
`x0_lambdas` init ∈ {0.0, 0.1, 0.3}; try freezing `resid_lambdas` at 1.0. **3 runs.**
→ Per-layer learned scalars are a small, poorly-understood part of the design.
Freezing them and losing nothing would be a simplification.

---

## Category F — Goal and harness meta-experiments

These change what "better" *means*. Each needs a different `AUTORESEARCH_GOAL` or
a new scorer in `objective.py`.

### F1. Memory-constrained ratchet
`AUTORESEARCH_GOAL=min_bpb_under_vram`, `AUTORESEARCH_VRAM_LIMIT_GB=8`. **~10 runs.**
→ You're at 15.5 GB. An 8 GB ceiling forces genuinely different choices. Compare
what a constrained search keeps against what the unconstrained one kept.

### F2. Parameter efficiency
`AUTORESEARCH_GOAL=min_bpb_x_params`. **~10 runs.**
→ Does quality-per-parameter select a different frontier? Given the embedding
imbalance, this may find B4/B5 on its own — which would be a nice validation that
the goal machinery does real work.

### F3. Goal disagreement ★ the interesting one
Take the top ~10 runs from any sweep and re-score them under all three goals
**offline** — no new runs, just `results.jsonl` and a few lines of pandas.
→ Do the goals rank differently? If all three agree everywhere, the goal machinery
is decoration. If they disagree, you've found where the trade-offs actually live.
**Zero GPU cost.**

### F4. Latency-aware scoring
Implement a decode benchmark and `min_bpb_x_latency` in `objective.py`
(`alternative-goals.md` §5.1). **~10 runs** after the plumbing.
→ There is no generation path in the repo at all. Train-time and inference-time
optima genuinely diverge, and this is where GQA (E1) and windows (E2) stop being
neutral.

### F5. Cheaper eval, more experiments
Set `AUTORESEARCH_EVAL_TOKENS` from A2's answer and run an otherwise identical
sweep. **~10 runs.**
→ Does a noisier metric with 30% more experiments find better configs per hour?
A direct test of the ratchet's noise/throughput trade.

### F6. Pareto frontier
Run the frontier cell in `analysis.ipynb` over everything accumulated, with
`Y_AXIS` ∈ {`peak_vram_gb`, `num_params_M`, `training_seconds`}.
→ Recovers trade-offs the scalar ratchet collapsed. Some frontier points will be
runs the ratchet *discarded*. **Zero GPU cost.**

---

## Category G — Deliberately interesting

Not expected to win. Each answers a question you can't get from a sweep.

### G1. No attention at all ★
Replace `self.attn(...)` with a no-op (`x = x + 0`). **1 run.**
→ How much is attention actually buying at 11.5M params on 152M tokens? If the
gap is small, everything in Category E is rearranging deck chairs and Category B
is where the wins are. One of the highest information-per-run experiments here.

### G2. Attention only
The mirror: remove the MLP. **1 run.**
→ With G1, decomposes the model's capability across its two halves.

### G3. Extreme aspect ratios
`DEPTH=16, ASPECT_RATIO=16` (deep and thin) vs `DEPTH=1, ASPECT_RATIO=1024`
(one enormous layer). **2 runs.**
→ Both probably bad. The *shape* of the badness tells you which axis the budget
is actually sensitive to.

### G4. Frozen random embeddings
Initialise `wte` randomly and exclude it from the optimizer. **1 run.**
→ Are 2.1M trained embedding parameters earning their place, or would a random
projection do? A surprising result here would redirect Category B entirely.

### G5. Single-shard training ★
Train on one data shard repeatedly instead of streaming. **1 run.**
→ Forces many epochs over little data. The train/val gap becomes visible in a way
it isn't at `epoch: 1`, which makes every regularisation experiment interpretable.

### G6. Equal tokens, opposite batch extremes
Two runs at matched total tokens: huge batch/few steps vs tiny batch/many steps
(adjust `TIME_BUDGET` to match token counts). **2 runs.**
→ Separates "more steps" from "more tokens" — normally confounded in every other
experiment in this document.

### G7. Reversed sequences
Reverse each row before training and eval. **1 run.**
→ Is English meaningfully harder backwards at this scale? Pure curiosity, and a
sanity check that the pipeline isn't accidentally order-invariant.

### G8. Vocabulary stress
Retrain the tokenizer at `VOCAB_SIZE` ∈ {2048, 4096} (needs `prepare.py` and a
fresh tokenizer; **not** ratchet-comparable). **2 runs + prep time.**
→ Directly attacks the embedding imbalance at its root. `val_bpb` is
vocab-size-independent by design, which is exactly what makes this comparison
legitimate where most `prepare.py` changes are not.

---

## Suggested first two nights

**Night 1 — calibration and the obvious mistunings** (~40 runs, ~4 hours)

1. A1 seed noise (5) — everything downstream depends on this
2. A3 determinism (2)
3. A2 eval noise (4) — may buy back time for the rest
4. C1 warmup (5) — most likely real win
5. B1 depth (5)
6. C7 batch size (4)
7. C4 matrix LR (5)
8. G1 no-attention (1) — cheapest big insight
9. B5 drop value embeddings (2)
10. B4 tied embeddings (2)

Then A4 budget transfer (4) on whatever won, before believing any of it.

**Night 2 — objectives and architecture**, chosen by what Night 1 found. If C1 or
C4 moved things, finish Category C first; the optimizer is upstream of everything.
If B4/B5 won, Category B's reinvestment variants come next. D1 and D2 are the
highest-value entries in Category D and both are unexplored on this hardware.

---

## What would make each category a success

- **A** — a number you trust, and a keep-threshold derived from it.
- **B** — knowing whether the embedding imbalance is real or a red herring.
- **C** — knowing which upstream hyperparameters didn't survive the move.
- **D** — one objective that beats plain CE at equal eval, or confidence that none does.
- **E** — one architectural simplification kept for free.
- **F** — evidence that goal choice changes the answer. If it doesn't, that's worth
  knowing before building more goal machinery.
- **G** — at least one result that changes what you'd try next.
