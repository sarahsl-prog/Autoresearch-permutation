# Experiment catalogue II — Kimi

A second catalogue of experiments for this machine and this harness. It assumes
the first catalogue (`experiment-catalogue.md`, categories A–G) and does not
repeat it — lettering continues at **H** so cross-references stay unambiguous.

Where the first catalogue was written before most of the runs existed, this one
is written *after* 45 of them. Everything below is grounded in what those runs
actually measured, and the most valuable categories here are the ones the data
forced on us.

---

## What the 45 runs in `results/results.jsonl` actually established

| # | Fact | Evidence |
|---|---|---|
| 1 | Init-noise floor σ ≈ 0.016–0.018 on `val_bpb` | A1, seeds 41–45: mean 1.3443, range 0.049 |
| 2 | That noise is **pure initialisation noise** | The dataloader has no RNG — shard order, packing and cropping are fully deterministic. Seed only reaches `init_weights` |
| 3 | Bitwise determinism holds **within a session** | A3: two identical runs, identical scores (1.3452, 1.3452) |
| 4 | **Cross-session drift is real and unmeasured** | Commit `3010f51`, seed 42, tb=300 scored **1.3411** (Aug 4), **1.3452** ×2 (Aug 5), **1.3541** (Aug 6). Same commit, same seed, three answers spanning 0.013 |
| 5 | Throughput drifts across sessions too | `fbf8e31`, seed 42, tb=300: **699 steps → 1.1957** (Aug 5), then **545 steps → 1.2138** (Aug 6). A 22% step-count swing on identical code |
| 6 | The ratchet's leaderboard is probably wrong | C7 batch 2¹⁷ scored **1.1837** and 2¹⁶ **1.1840** — both *better* than the kept best (`B1 depth=3`, 1.1957) — yet both were discarded |
| 7 | Value embeddings are load-bearing | B5: dropping them cost **+0.19 bpb** (1.3841), the largest single regression measured |
| 8 | Tied embeddings fail *hard*, then partially recover | B4: 2.7083 → 2.6434 → 1.2309 → 1.1946 as fixes landed. Cause never localised |
| 9 | No-attention diverges; the number is still missing | G1 NaN'd twice on-protocol. Off-protocol diags say ≈1.85. The catalogue's highest-information experiment has no valid result |
| 10 | Budget transfer holds | A4: the winner gained −0.064 from a 3× budget while the baseline gained −0.010. Rankings at 300s mean something at 900s |
| 11 | The current best is 78% embeddings by parameter | `fbf8e31`: 10.75M params = 8.39M embeddings (wte 2.1M + lm_head 2.1M + VE 4.2M) + 2.36M transformer |
| 12 | Warmup=0.01 beat warmup=0 — but the comparison was confounded | See N1: the first ~12 steps are untimed **and** run at lrm=0 when `WARMUP_RATIO>0`, but at full LR when it is 0 |

**The noise budget.** With σ_init ≈ 0.017 (fact 1) and session drift of
~0.013–0.018 (facts 4–5), a single-run delta below ~0.03 is not a decision —
it is a coin flip with extra steps. Most of category C's sweep results live
inside that band. Every category below inherits this: **replicate before you
believe**, and prefer comparisons that can be run back-to-back in one session.

**Cost model:** unchanged — 5.8 min/run, ~10 runs/hour, ~83 runs/night.
This catalogue is ~85 runs: more than one night. Order by information density.

---

## Protocol additions

The first catalogue's seven rules still hold. Three more, forced by the facts
above:

8. **A/B inside one session.** Fact 4 means two configs compared across days
   carry an extra ~0.013 of noise. Wherever a comparison matters, run the
   variants back-to-back in the same session, interleaved (A,B,A,B) if seeds
   are involved. Intra-session determinism (fact 3) makes same-session
   comparisons exact.
9. **Record `num_steps` as a first-class outcome.** Fact 5: step count is a
   property of the *session*, not just the config. A run with 20% fewer steps
   is a different experiment, and `results.jsonl` already carries the column —
   use it before comparing scores.
10. **A keep is a hypothesis, not a result.** Any kept config whose margin is
    under ~2σ goes on the rematch list (H2), not on the pedestal.

---

## Category H — The ratchet audit: are the keeps real?

The first catalogue calibrated the *metric* (A1–A6). Nobody has calibrated the
*decisions*. Facts 4–6 say the ratchet has been keeping and discarding inside
its own noise. These runs measure the decision-making itself.

### H1. Noise floor at the current operating point
`fbf8e31` (depth 3, warmup 0.01), seeds {41,42,43,44,45}. **5 runs.**
→ A1 measured the old depth-4 baseline. Noise may scale with depth, step count,
or loss level, and every keep-threshold in this catalogue is keyed to it. If σ
at 1.19 is much smaller than at 1.34, the bar for future keeps is lower than we
think; if larger, half the existing keeps are suspect.

### H2. The rematch ★ highest value in this document
Head-to-head, 3 seeds each, interleaved in one session: `fbf8e31` (kept,
1.1957) vs C7-2¹⁷ (discarded, 1.1837) vs C7-2¹⁶ (discarded, 1.1840) vs
B5-reinvest (discarded, 1.1895). **12 runs.**
→ The single-run ranking says the ratchet kept the *wrong* config. Note the C7
runs needed `DEVICE_BATCH_SIZE` lowered to satisfy the divisibility assert, so
they were never one-knob changes — the rematch should keep that and say so.
Either outcome is valuable: confirm the current best, or promote a config the
ratchet threw away and learn the ratchet's false-negative rate directly.

### H3. Session drift ★
Same commit, same seed, across controlled session changes: warm boot, cold
boot, after clearing the torch/cuBLAS caches, busy vs idle machine. Log
`nvidia-smi` clocks at startup. **4 runs.**
→ Fact 4 (1.3411 → 1.3541 at equal step counts) says numerics themselves drift
across sessions — prime suspects are cuBLAS algorithm selection and compile
caches, not the GPU's math. Fact 5 (699 → 545 steps) says throughput drifts
too, and under a wall-clock budget throughput *is* score. These two need
separating: add a **numerics canary** to `train.py` — a few lines that hash a
fixed small computation at startup — so every future run carries a fingerprint
of its numeric environment. If the canary is stable while scores drift, it's
throughput; if it flips, it's numerics.

### H4. Ratchet optimism
**0 extra runs.** From H1–H3's data: E[kept score − replication mean] is the
ratchet's selection bias — the amount by which every kept score overstates its
config. `fbf8e31` already shows 1.1957 → 1.2138 on re-measurement (+0.018).
→ This number belongs at the top of every future results table, next to σ.

### H5. Data-axis noise
Move shards 0–9 out of `DATA_DIR`, drop in shards 10–19 (a file operation — no
code changes, `prepare.py` untouched), run the current best twice. **2 runs.**
→ A1 measured init noise; H3 measures session noise; *which ~2.5 shards the
deterministic stream happens to contain* has never been isolated. The training
set is ~610M tokens of which a run sees ~150M — the slice is arbitrary and
frozen. If this axis is large, every result in both catalogues is partly a
property of shards 0–2.

---

## Category I — Instrumentation before expenditure

The next ~70 runs produce numbers. These make the numbers explainable. Cheap
enough that skipping them is false economy.

### I1. Gradient telemetry ★
Log per-group grad norm and update RMS (Muon groups vs AdamW groups) every 10th
step via the buffered `tracking.log_step` path — kept outside the timed window
per the harness contract; every-10th keeps the tax ~nil. **1 run** at the
current best.
→ Defines what "healthy" looks like at this scale. Makes G1's NaN, C1's warmup
effect, and any future divergence diagnosable after the fact instead of
mysterious. Also answers whether Muon and Adam updates stay commensurate as the
schedule anneals.

### I2. Per-position loss map ★
One-off diagnostic (a script, not a ratchet run): per-position eval bpb across
the 2048 positions of a trained model.
→ The packer is BOS-aligned, so position 0 is always a document start and
cross-document boundary positions are identifiable. Expect a U-shape; the
*size* of the early-position dip bounds what D4-style position weighting could
ever recover, and the cross-boundary share of total loss bounds the upside of
K1 before spending its implementation cost. Do this **before** K1.

### I3. Packing forensics
Pure CPU, zero GPU: instrument the packer offline — crops per row, documents
per row, doc-length vs position correlation, buffer turnover rate.
→ Best-fit packing with a 1000-doc buffer has never been measured. It prefers
long docs early in each row and crops the shortest when stuck, so position and
document length are correlated by construction. K8 and D4 both manipulate this
structure; know it first.

### I4. Attention-map autopsy
One trained run with attention weights captured at eval: mass across document
boundaries, mass on BOS (sink behaviour), per-head entropy. **1 run.**
→ Direct evidence for the questions K1 and K5 ask indirectly. If 2-head
attention already ignores cross-boundary context, K1's masking is free; if it
leans on it, K1 will hurt and that's a finding about what packing teaches.

---

## Category J — Optimizer internals

Category C swept the LRs and betas. It never opened the optimizer. MuonAdamW
has four load-bearing mechanisms beyond its learning rates, and none has been
ablated at this scale.

### J1. Gradient clipping ★
There is none. Global-norm clip at {1.0, 0.5}. **2 runs.**
→ ~580 steps at high LR with no clip: a handful of spike steps could be costing
real bpb, and I1's telemetry will say. Also cheap insurance for every other
experiment in this catalogue — several crashes in the log might never have
happened.

### J2. Muon for the unembedding
`lm_head` is an 8192×256 matrix trained by Adam at 0.004. Move it to a Muon
group (mind the ×5.66 shape-scaling Muon applies to tall matrices — sweep its
LR down to compensate). **2 runs.**
→ The orthogonalisation Muon provides may be exactly what a 2.1M-param output
projection wants — or exactly what it doesn't. Either way you learn whether
Muon's value is *where* it's applied, not just *that* it's applied.

### J3. The deliberate swap
Adam for the transformer matrices, Muon for the embeddings. **1 run.**
→ The mirror of J2, asked adversarially. If this is much worse, Muon's
placement on matrices is the whole story; if it's fine, the conventional
wisdom about which params need orthogonalisation is wrong at this scale.

### J4. NorMuon ablation
Remove the second-momentum variance reduction from `muon_step_fused` (keep the
polar-express orthogonalisation). **1 run.**
→ NorMuon was tuned upstream at 50M+ params. At 2.4M matrix params it may be
ballast. Equal-or-better is a simplification keep under the simplicity
criterion.

### J5. Cautious-mask ablation
The weight decay is masked by `(g · p) ≥ 0` ("cautious"). Remove the mask —
plain decoupled decay. **1 run.**
→ C6 sweeps the decay *magnitude*; nobody has asked whether the mask itself
matters at 580 steps. One run settles it.

### J6. Adam epsilon
`eps` ∈ {1e-8, 1e-6}, from 1e-10. **2 runs.**
→ With `UNEMBEDDING_LR=0.004` and tiny gradients late in warmdown, eps is a
hidden LR floor on the unembedding. C10 swept betas; eps is the other
denominator knob and it's never been touched.

### J7. Asynchronous schedules
Embeddings hold peak LR while matrices anneal on the normal schedule; and the
mirror. **2 runs.**
→ One schedule serves a model that is 78% embeddings by parameter (fact 11).
The embedding table's loss landscape and the transformer's are different
objects; annealing them together is an assumption, not a law. Interacts with
everything category C found.

### J8. Nesterov off
Plain heavy-ball momentum in Muon (drop the lookahead lerp). **1 run.**
→ C8 varied the momentum *ramp*; the Nesterov itself is unexamined. At 580
steps the lookahead horizon may be miscalibrated.

---

## Category K — Context structure: what the model is allowed to see

E2 varied windows *across layers*. D5 annealed sequence *length*. Nobody has
touched what attention can reach *within* a packed row, or whether positions
are encoded the right way at all.

### K1. Document-boundary masking ★
Block attention across document boundaries: detect BOS positions in `idx`
(legal — `forward` owns its masks), build per-row document IDs, mask
cross-document attention. Eval rows are BOS-packed too, so train and eval stay
consistent. **2 runs** (masked vs not, same session).
→ Cross-document bleed is either free context or pure noise, and modern
pipelines disagree about which. At 152M tokens this is a data-efficiency
question with a real answer. **Implementation honesty:** on the SDPA backend
this costs the `is_causal` fast path — FlexAttention with a captured doc-id
tensor computes densely, so report `num_steps` alongside bpb. Run I2 first: if
cross-boundary positions are a negligible share of eval loss, the upside here
is capped and the run is purely diagnostic.

### K2. Train short, eval long
Reshape every batch `(B, 2048) → (4B, 512)` for the *entire* run — no anneal
(that's D5). Eval stays at 2048. **1 run.**
→ Attention is ~57% of per-token FLOPs at 2048 in this config, so this buys
~1.7× throughput, spent as more steps. The price is a 4× rotary extrapolation
at eval. The run measures length generalisation *and* banks a large throughput
win if rotary holds — and "does rotary hold at 4×" is a fact worth having
regardless of the score.

### K3. NTK-scaled rotary
Rebuild the cos/sin buffers with NTK-by-scaling for the eval length, paired
with K2. **1 run.**
→ The buffers are built in `train.py`, so this is a legal architecture change
that affects train and eval identically — no eval-contract violation. If K2
fails and K3 recovers it, the failure was rotary's, not the model's.

### K4. Multi-scale heads
Per-head windows *within* a layer: half the heads at 512, half full-context,
via an H-indexed BlockMask. **2 runs.**
→ E2 varied windows across layers; the per-head axis (Mistral/Longformer-style
multi-scale) is orthogonal and untouched. At `n_head=2` it's one local + one
global head — maximally legible.

### K5. NoPE
Delete rotary entirely. Keep QK-norm. **1 run.**
→ Causal attention alone encodes position surprisingly well at this context
length. If the delta is small, rotary is 20480×64×2 of buffer and two kernels
per layer bought for nothing — a simplification candidate, and it changes what
K2/K3 mean.

### K6. Partial rotary
Rotate half the head dimensions, leave half position-free (NeoX-style
`partial_rotary_factor=0.5`). **1 run.**
→ The interpolation between E6/K5: some channels carry position, some carry
pure content. Cheap, and the answer localises *where* position information
actually lives in the head.

### K7. Window curriculum
All-S early, anneal to all-L by mid-run. **2 runs.**
→ D5 reshapes tokens; this reshapes attention span at full sequence length.
Cheap attention early buys steps when gradients are noisy anyway; full context
arrives once there's something worth attending to. Orthogonal to D5 and
combinable with it later if both win.

### K8. Buffer size
`make_dataloader(..., buffer_size=)` ∈ {100, 10000}, from 1000. **2 runs.**
→ Exposed in the signature — no `prepare.py` edit needed. The buffer is the
packer's search pool and the stream's only shuffle-like mechanism: small
buffer = more crops and less local diversity, large = the opposite. I3 says
what the baseline does; this says whether it matters.

### K9. Input token dropout
Replace ~5% of *input* tokens with `reserved_1` (unused; targets untouched).
**1 run.**
→ Regularisation aimed at the 78%-embedding model: does it over-rely on exact
local token identity? Gate on A5's train/val gap — if there's no overfitting
signal at all, expect this to hurt, and skip it in favour of L-category runs.

---

## Category L — Parameter budget, round 2

Category B redistributed parameters on theory. Three of its results came back
with *measured* surprises (facts 6–8), and those deserve follow-ups that
localise causes rather than sample neighbours.

### L1. The tying autopsy ★
B4 went 2.71 → 1.23 → 1.19 as fixes landed, cause unknown. Prime suspect:
`EMBEDDING_LR=0.6` vs `UNEMBEDDING_LR=0.004` — a 150× gap that tying forces
into one LR. Run tied embeddings with **per-role gradient scaling** (scale the
wte-role and lm_head-role gradient contributions to match their untied update
magnitudes). **2 runs.**
→ If gradient scaling recovers most of the gap, B4's failure was an LR
mismatch and tying is still viable — a real parameter saving back on the
table. If not, it's capacity, and the embedding imbalance hypothesis (fact 11)
survives intact. Read B4's MLflow curves first (P2): divergence vs slow start
are different stories.

### L2. Factored embeddings
ALBERT-style: `wte = Embedding(8192, 128) @ Linear(128, 256)`, same for
`lm_head` (separate factors). Saves ~2M params without tying's rigidity;
reinvest in width or depth. **2 runs.**
→ B4 attacked the imbalance by sharing, B5 by deleting, B6 by low-ranking the
VE. Factoring is the fourth distinct attack and the only one that keeps the
input and output embeddings independent *and* cheap.

### L3. Value-embedding placement
`has_ve` is alternating-plus-last. Try first-layer-only, last-layer-only,
all-layers. **3 runs.**
→ B5 proved VE matters (+0.19 bpb — fact 7) but not *where*. At 4.2M params
(39% of the model), which layers hold VE is the biggest unexamined placement
decision in the architecture.

### L4. Layer tying
One block, applied 3× (keep per-layer `resid_lambdas`/`x0_lambdas` and the
per-layer VE tables). Saves ~1.6M transformer params; reinvest. **2 runs.**
→ The ALBERT question at 11.5M scale, and a sharp test of what depth is for at
depth 3. Note the trap: reinvestment mostly feeds embeddings (fact 11), so the
run also measures whether *any* transformer-param saving can be usefully
reinvested in this regime.

### L5. Learnable norm gains
`norm()` is parameter-free RMSNorm — unusual; every production LLM has
per-channel gains. Add them (keep QK-norm gain-free). **1 run.**
→ One run answers whether the model wants per-channel rescaling capacity it
currently lacks. Equal-or-worse is a nice simplification confirmation.

### L6. VE gate width
`ve_gate` reads `x[..., :32]`. Try {8, 128}. **2 runs.**
→ How much of the hidden state does the value-embedding gate need to see?
Micro, but it probes the information flow of the single most valuable
component measured (fact 7).

---

## Category M — The train/eval divergence you choose

Eval currently scores the final, raw, soft-capped model. Each of those three
choices can be broken on purpose, legally — `train.py` owns its weights and its
training objective; only `forward`'s contract is fixed.

### M1. EMA-as-model ★
Maintain an EMA of the weights (decay {0.98, 0.995} — half-lives of ~35 and
~140 steps against a ~580-step run), load it into the model before
`objective.report`. **2 runs.**
→ Not D6 (EMA-as-*teacher*): this changes *which weights get scored*. Under
short budgets with annealed LRs, the EMA is often simply better than the final
point. If it wins here it's close to free — a few lines and one extra
parameter set — and it reframes what warmdown is for (M3).

### M2. LAWA
Average the last ~10% of step snapshots, eval that. **1 run.**
→ The cheaper cousin of M1 (no per-step EMA cost, just K stored states). If
LAWA ≈ EMA, take the simpler one.

### M3. Averaging vs annealing
`WARMDOWN_RATIO≈0` (constant LR to the wall) + M1's EMA eval, vs baseline
warmdown + raw eval. **2 runs.**
→ How much of the warmdown dividend is just weight averaging in disguise?
C2/C3 swept the schedule shape; this decomposes *why* it works. If EMA
captures it, the schedule knob gets simpler forever.

### M4. Dual eval
Once M1 lands: eval raw *and* EMA in every run (+~40s). **0 extra runs.**
→ The per-config EMA delta is itself a measurement — configs with a large delta
are under-annealed; configs with none are converged. Telemetry, not just score.

### M5. Asymmetric softcap
Train on uncapped logits (`training_loss` computes its own, pre-softcap), keep
the softcap on the eval path. **1 run.**
→ E7 sweeps the cap on both paths; D2 replaces it with z-loss; nobody has
*split* them. If training uncapped while evaluating capped is neutral-or-better,
the cap is an eval-time crutch the model doesn't need during learning — and
that says something real about where logit explosion pressure comes from.

---

## Category N — Schedules, and the untimed-steps confound

### N1. The free-steps confound ★
The time budget excludes the first ~12 steps (`step > 10` gate). With
`WARMUP_RATIO>0`, those steps run at **lrm=0** — optimizer state warms,
parameters frozen. With `WARMUP_RATIO=0`, they run at **full LR** — free,
untimed training. C1 compared exactly these two things and called the
difference "warmup". Quantify it: baseline vs a schedule whose warmup clock
starts when the budget clock does. **2 runs.**
→ Part of C1's 0.04-bpb win may be schedule bookkeeping, not warmup. Knowing
how much matters for every schedule conclusion drawn so far — and the human
may want to close the loophole (count from step 0, or harden the wall-clock
cap) once its size is known.

### N2. Step-based schedules
Probe steps/sec for ~10s, estimate total steps, schedule on
`step / estimated_total` instead of seconds. **2 runs.**
→ Fact 5: throughput swings ~20% across sessions. Time-based schedules
self-adjust the *shape* but not the *step count*; step-based schedules fix the
shape in optimizer space and let wall clock float. Which one is robust to a
noisy machine is a real question with a measurable answer.

### N3. Warm restarts
Two cosine cycles inside the 300s. Pair the cycle endpoints with M2-style
averaging — a two-model soup from a single run. **2 runs.**
→ SGDR at micro-scale. Restarts test whether the loss landscape at this budget
has multiple basins worth visiting; the soup tests whether they're connected.
Two classical questions, one cheap experiment.

### N4. Batch growth
2¹⁶ tokens/step for the first third (slice the loader's rows — 3 lines), 2¹⁸
after. **2 runs.**
→ The dynamic version of C7 — and C7's *discarded* runs (1.1837, fact 6) say
small batches were already competitive. More steps early when gradients are
noisy, less gradient noise late when fine-tuning the descent. If H2 promotes
2¹⁷ outright, this becomes the refinement instead.

### N5. The missing endpoint
C2 swept `WARMDOWN_RATIO` ∈ {0.2, 0.35, 0.5, 0.7} — never 0.0. Constant LR to
the wall, scored with M1's EMA eval. **1 run.**
→ Completes the sweep's range and feeds M3's decomposition. Without the 0.0
endpoint the warmdown curve's shape is extrapolated, not measured.

---

## Category O — Throughput is quality

Under a fixed wall-clock budget, systems choices *are* hyperparameters. Fact 5
proves the coupling is large. These are the honest ways to exploit it.

### O1. max-autotune ★
`torch.compile(mode="max-autotune")`. **1 run.**
→ Compilation happens during the untimed first steps, so its extra cost is
largely free under the current budget accounting — a legitimate exploit worth
flagging as such (see "harness notes" below). Autotune caches are also a
session-drift suspect, so this run doubles as an H3 data point.

### O2. bf16 logits
The `.float()` + softcap path materialises ~8.6 GB of fp32 logits per copy at
128×2048×8192 — the bulk of the 12.7 GB peak. Compute CE in bf16 (or chunked
fp32), reinvest the headroom in batch or width. **2 runs.**
→ Does logit precision move bpb at all? If not, several GB come back for free
and every memory-limited config in both catalogues gets re-evaluated.

### O3. Gradient checkpointing
Recompute activations, buy a wider or deeper config. **1 run.**
→ Usually a loss under a time budget (you pay steps for capacity), which is
exactly why one run suffices: it prices the exchange rate directly.

### O4. Accumulation vs device batch
64×2048×2accum vs 128×2048×1 — identical math, different kernels. **1 run.**
→ Isolates pure kernel efficiency from optimization. Whatever delta appears is
the accumulation overhead, measured cleanly, and it sizes the free win
available to any config that needs a smaller device batch for memory reasons.

---

## Category P — Rescue missions

Three failures in the log still have information in them. Each gets one
careful, on-protocol attempt.

### P1. G1 rescue ★
No-attention NaN'd twice on-protocol; off-protocol diags say ≈1.85. One clean
run at the standard budget with J1's clipping and a reduced `MATRIX_LR`.
**2 runs.**
→ The first catalogue called no-attention its highest-information experiment
and it still has no valid number. "How much is attention worth at 11.5M params
on 152M tokens" deserves a real answer, not a crash log.

### P2. B4 forensics
Before spending GPU: read the four tied-embedding runs' MLflow curves.
Divergence, slow start, or late instability each point at a different cause —
and at a different fix for L1. **0–1 runs.**
→ Four crashed/failed runs already exist; their curves are free.

### P3. C7 reproduction
Reproduce 2¹⁷ with full telemetry (I1 in place) and the two-knob change
(documented `DEVICE_BATCH_SIZE` adjustment) made explicit. **1 run.**
→ Either it replicates and H2 was urgent, or it doesn't and the log's most
confusing entry gets an explanation.

---

## Category Q — Zero-GPU investigations

Pure analysis. Do these during compile time.

### Q1. The early-abandonment rule ★
Across the 45 existing runs: how well does train loss at 60s/120s
rank-correlate with final `val_bpb`? If Spearman ρ is high, the loop can kill
the bottom half of runs at a third of the budget — **~2× experiment throughput,
permanently.**
→ The single highest-leverage item in this document, and it costs no GPU.
Needs the per-step curves from MLflow; if the server has lost them, add a
local `curves.jsonl` artifact first (`tracking.py` is read-only to the agent,
not to you) so the question stays answerable.

### Q2. Curve-shape taxonomy
Spike counts, plateau onset, warmdown slope — do kept runs' curves look
different from discards'? → If yes, curve shape becomes a cheap pre-eval
screen; if no, final-score noise really is irreducible from the curve.

### Q3. The `3010f51` forensics
One commit + one seed produced 1.3411, 1.3452, 1.3541 (fact 4). Before H3
spends runs: diff the environment across those timestamps — caches, driver
state, step counts (578 vs 579 vs 575 — too small to explain 0.013).
→ Pure archaeology, and it decides whether H3 is measuring numerics drift,
throughput drift, or an unrecorded config difference.

### Q4. The iso-token view
Replot every completed run as bpb vs **tokens seen** (`num_steps × batch`)
instead of vs config. → `fbf8e31`'s three points (142.9M→1.2138,
183.2M→1.1957, 504.6M→1.1494) should collapse onto one smooth curve if token
count is the whole story; the C7 and B1 sweeps test whether *other* configs
land on the same curve. Whatever doesn't fit the curve is a real effect;
whatever does was never an effect. This may retroactively explain half the
scatter in the log.

---

## Suggested first two nights

**Night 1 — audit the ratchet (~24 runs, ~2.5 h)**
1. Q3, Q4, Q1 (0 GPU — do during the first compiles)
2. H3 session drift (4) — with the numerics canary added
3. H1 noise floor at the current best (5)
4. I1 telemetry baseline (1)
5. H2 the rematch (12) — interleaved, one session
6. H5 data axis (2)

Ends with: a noise model that survives replication, a corrected leaderboard,
and a measured selection bias for every keep decision ever made here.

**Night 2 — mechanisms (~20 runs, ~2 h)**
Ordered by Night 1's keep-threshold: J1 clip (2), M1+M4 EMA (2), N1 the
confound (2), I2 position map then K1 doc-masking (1+2), O1 max-autotune (1),
O2 bf16 logits (2), L1 tying autopsy (2, after P2's curve read), P1 rescue (2),
M3/N5 annealing decomposition (3).

---

## What would make each category a success

- **H** — a noise model that replicates, a corrected leaderboard, and a
  published optimism number for the ratchet. Even "the keeps were all fine" is
  a result the first catalogue assumed and never tested.
- **I** — no future crash or spike is ever *mysterious* again.
- **J** — knowing which of MuonAdamW's four mechanisms are load-bearing at
  this scale, and a clip value that makes the whole loop safer.
- **K** — knowing whether document boundaries, rotary, and window structure
  are helping, hurting, or free.
- **L** — a causal story for B4 and B5, not just their scores.
- **M** — a near-free ~0.01–0.02 from averaging, or certainty that none
  exists, plus a decomposition of what warmdown actually does.
- **N** — schedules robust to a noisy machine, and C1's win correctly
  attributed.
- **O** — more steps per run without touching the math, or proof the ceiling
  is elsewhere.
- **P** — three failures converted into numbers.
- **Q** — an early-abandonment rule that permanently doubles the loop's rate.

---

## Further afield — noted, not sized

Ideas that came up while building this catalogue and may prove fruitful later,
in rough order of expected value per effort:

- **Harness hardening.** N1 and O1 both exploit the untimed first-steps window.
  Once their size is measured, the human (not the agent) may want to close the
  loophole: count all steps from step 0, or cap total wall clock rather than
  training time. Worth deciding *before* an autonomous night, not after.
- **The numerics canary** (from H3) deserves to be permanent: a startup hash
  of a fixed computation in every run's record makes session drift visible
  forever, for free.
- **A local per-step curve artifact** (`curves.jsonl` next to `run.json`) so
  Q1/Q2-style analysis never depends on the MLflow server's retention.
- **Learned null-KV attention sink** (one extra learned k/v per head so
  softmax can attend to nothing): expressible via FlexAttention's `score_mod`;
  interesting precisely because `n_head=2` gives the model so few places to
  dump attention.
- **Cross-layer KV sharing** at depth 3 (layers share K/V, compute once):
  parameter and throughput savings aimed at the inference goals in
  `alternative-goals.md` §5.1.
- **Differential or sigmoid attention**: need custom kernels the SDPA path
  can't express cheaply — revisit only if K-category results say attention
  internals are where the remaining loss lives.
- **Stochastic depth**: probably premature at depth 3, but the natural
  regularisation counterpart to K9 if A5 ever shows a train/val gap.
- **Token-budget replication of the headline results** (once
  `alternative-goals.md` §4.1 lands): H2's winner under a fixed *token* budget
  instead of wall clock would separate "better config" from "faster config" —
  the confound at the heart of fact 5.
- **A second pinned val shard used only for confirmation** (§6.1 of
  `alternative-goals.md`): after H-category quantifies selection bias, a
  held-out shard measures how much of it is val-overfitting specifically.
