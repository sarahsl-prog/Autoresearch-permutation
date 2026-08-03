# Training models with different goals

Brainstorming notes on how this repo could be modified so the autonomous loop
optimizes for something other than "lowest `val_bpb` in 5 minutes of wall clock".

Nothing here is guaranteed to work. Everything is meant to be cheap enough to
try, and each item names the specific lines that would have to change.

---

## 1. Where the goal actually lives today

The DataCamp guide describes the repo as three files with three roles: `prepare.py`
is the neutral judge, `train.py` is the sandbox the agent rewrites, and `program.md`
is the only file the human writes. That framing is useful, but the goal is not
actually contained in one place — it is smeared across four:

| Layer | Location | What it fixes |
|---|---|---|
| **Metric** | `prepare.py:343-365` (`evaluate_bpb`) | *How* quality is measured (bits/byte on a pinned val shard) |
| **Budget** | `prepare.py:31` (`TIME_BUDGET`), enforced at `train.py:603` | What resource is held constant (wall clock) |
| **Training objective** | `train.py:287-290` (`F.cross_entropy` inside `forward`) | What the model is actually asked to learn |
| **Selection rule** | `program.md:33`, `program.md:103-104` | How results turn into keep/discard decisions (the ratchet) |

Changing the goal means changing one or more of these four. They are largely
independent, which is the good news: most of the ideas below touch exactly one.

### 1.1 The single most important structural problem

`evaluate_bpb` does not compute its own loss. It calls back into the model:

```python
# prepare.py:359
loss_flat = model(x, y, reduction='none').view(-1)
```

…and `GPT.forward` computes the loss itself (`train.py:287-290`), including the
logit softcap at `train.py:282-285`. So **the "fixed, read-only metric" is
partially implemented inside the file the agent is allowed to rewrite.**

Today this is mostly harmless because the only sane loss is cross-entropy. The
moment you introduce a different *training* objective (label smoothing, z-loss,
focal loss, auxiliary heads), an agent editing `forward` will silently change the
number that decides whether its own experiment is kept. That is not adversarial
reward hacking — it is an honest agent breaking the scoreboard by accident.

**Prerequisite refactor for almost everything below:** split the model's forward
pass from its loss.

```python
# train.py — replace the tail of GPT.forward
def forward(self, idx, targets=None, reduction='mean'):
    ...
    logits = self.lm_head(x)
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    if targets is None:
        return logits
    # eval-facing path: plain CE, kept deliberately boring
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                           ignore_index=-1, reduction=reduction)

def training_loss(self, idx, targets):
    """Agent is free to make this arbitrarily weird."""
    return self.forward(idx, targets)
```

The training loop at `train.py:548` calls `model.training_loss(x, y)`; `evaluate_bpb`
keeps calling `model(x, y, reduction='none')`. Now the objective and the metric
can diverge on purpose, which is the whole point of section 3.

### 1.2 The second structural problem: the judge doesn't measure anything

Every number the ratchet consumes is self-reported by the file under test:

- `total_training_time` is accumulated by `train.py:579`
- `peak_vram_mb` reads `torch.cuda.max_memory_allocated()` at `train.py:619`
- `num_flops_per_token` comes from `estimate_flops` at `train.py:208`
- the summary block itself is a `print` at `train.py:621-630`

Under the current goal this barely matters — you can't fake your way to a lower
bpb by mis-timing yourself, you can only give yourself more training. But **any
goal that scores a resource (memory, FLOPs, latency, tokens) puts that resource's
meter inside the sandbox.** If you want to optimize FLOP efficiency, `estimate_flops`
has to move out of `train.py` first.

General principle worth adopting before adding any new goal: *the judge measures,
the sandbox trains.*

---

## 2. Scaffolding to add first

These aren't goals, they're the plumbing that makes goal-swapping a config change
instead of a rewrite. Maybe half a day of work total.

### 2.1 `objective.py` — a second read-only file

```python
# objective.py  (read-only, same status as prepare.py)
from prepare import evaluate_bpb

GOAL = "min_bpb_fixed_time"   # the only line a human edits

def score(model, tokenizer, batch_size, stats):
    """Returns (scalar_to_minimize, dict_of_metrics). stats = telemetry from the run."""
    bpb = evaluate_bpb(model, tokenizer, batch_size)
    metrics = {"val_bpb": bpb, **stats}
    if GOAL == "min_bpb_fixed_time":
        return bpb, metrics
    if GOAL == "min_bpb_under_20gb":
        penalty = float("inf") if stats["peak_vram_mb"] > 20 * 1024 else 0.0
        return bpb + penalty, metrics
    if GOAL == "min_bpb_x_latency":
        ms = benchmark_decode(model, tokenizer)
        metrics["ms_per_token"] = ms
        return bpb * (ms ** 0.25), metrics
    ...
```

`train.py:610-613` becomes `score, metrics = objective.score(...)`, and `program.md`
stops naming `val_bpb` — it says "minimize the `score:` line". Swapping goals is
then a one-line edit to a file the agent can't touch.

### 2.2 Structured results instead of grepped prints

`train.py:621-630` prints a fixed block that `program.md:61` greps. Emit JSON
alongside it:

```python
import json
with open("run.json", "w") as f:
    json.dump({"score": score, **metrics}, f)
```

Multi-metric goals (section 4) are miserable to express through `grep "^val_bpb:"`.

### 2.3 `results.tsv` → `results.jsonl`

The 5-column schema at `program.md:68-72` hard-codes one metric. A JSON-lines log
with arbitrary keys costs nothing, keeps `analysis.ipynb` easy to adapt, and lets
you compute a Pareto frontier after the fact even if the live ratchet is scalar.

---

## 3. Different training objectives (same evaluation)

These change *what the model learns* while leaving `evaluate_bpb` alone, so results
stay comparable to the existing baseline. Requires the §1.1 refactor. This is the
highest-value cluster to try first, because the comparison stays honest for free.

### 3.1 Multi-token prediction

Add `k` auxiliary heads predicting tokens `t+2 … t+k`, train on the sum, evaluate
on next-token only (Gloeckle et al. / DeepSeek-V3 style). The aux heads are pure
training-time scaffolding and get ignored at eval.

```python
# in GPT.__init__
self.aux_heads = nn.ModuleList([nn.Linear(n_embd, vocab_size, bias=False)
                                for _ in range(K - 1)])

# in training_loss
loss = self.forward(idx, targets)
for j, head in enumerate(self.aux_heads, start=2):
    shifted = targets[:, j-1:]              # predict t+j
    loss = loss + AUX_WEIGHT * ce(head(x[:, :-(j-1)]), shifted)
```

Cheap, well-precedented, and interacts interestingly with the fixed time budget —
the aux heads cost throughput, so it's a real trade rather than a free lunch.
Sweep `AUX_WEIGHT` and `K`.

### 3.2 Loss shaping

A family of one-line experiments, all now safe because eval is decoupled:

- **z-loss** (`+ λ·log²Z`) — likely interacts with the existing softcap at
  `train.py:282`; possibly *replaces* it, which would be a simplification win under
  the criterion at `program.md:37`.
- **label smoothing** — usually hurts bpb, but might help under a short budget.
- **entropy bonus / temperature on the training target** — cheap to try.
- **per-position loss weighting** — the dataloader packs documents BOS-aligned
  (`prepare.py:276-337`), so early positions in each row are systematically easier.
  Downweighting them changes the effective gradient a lot for ~3 lines of code.

### 3.3 Sequence-length curriculum

Rotary embeddings are precomputed to `10 × sequence_len` (`train.py:144`) and FA3
handles variable `T`, so the model already supports training at shorter contexts.
Reshape the loader output inside `train.py` — `(B, 2048)` → `(4B, 512)` — early in
training, anneal to full length. Same tokens/step, much cheaper attention, so it
buys steps under the fixed time budget. **This is a data-side experiment that does
not require touching `prepare.py`**, which is a general trick worth calling out: the
loader hands you tensors at `prepare.py:337`, and `train.py` may rearrange them
however it likes.

### 3.4 Fill-in-the-middle / prefix-LM mixing

Same trick as above — permute spans within `x` in `train.py`, add sentinel tokens
(there are 3 unused reserved tokens, `prepare.py:50`). Note this one *will* move
`val_bpb` in the wrong direction by construction; it only makes sense paired with a
goal change from section 5 (e.g. scoring infilling ability).

### 3.5 EMA self-distillation

No teacher checkpoint is available and 5 minutes is not enough to train one, but a
slow EMA copy of the model as a teacher costs one extra parameter set and a forward
pass. Consistency loss against the EMA's soft targets. Plausible under short budgets
where the model is far from convergence; may just be an expensive LR schedule.

---

## 4. Different resource budgets

Right now the loop holds *wall clock* constant (`prepare.py:31`, `train.py:603`).
This is what makes results non-portable across GPUs, as the README notes, and it
means the agent is partly optimizing MFU rather than modeling. Change the budget and
you change the entire character of what gets discovered.

### 4.1 Fixed token budget

Stop after N tokens instead of 300 seconds. The agent stops being rewarded for
kernel-level throughput tricks and starts being rewarded for sample efficiency —
a genuinely different research direction, and one whose findings transfer to other
hardware. Implementation is a two-line change at `train.py:603`, plus a wall-clock
*cap* (say 15 min) so a pathological config can't stall the overnight loop.

### 4.2 Fixed FLOP budget

Use `estimate_flops` as the meter — but move it to the read-only file first (§1.2),
otherwise "reduce FLOPs" has a trivial and very tempting wrong answer. More
principled than tokens for comparing architectures of different widths.

### 4.3 Time-to-target

Invert the problem: minimize seconds to reach `val_bpb ≤ 1.05`. Requires periodic
eval, which doesn't exist today (there's only the final eval at `train.py:610-613`).
Add a cheap mid-training eval on a small token count, with the expensive
`EVAL_TOKENS` check reserved for confirmation. Produces very legible results
("we went from 240s to 180s") and naturally rewards warmup/schedule work. Watch out:
the eval itself now costs budget, so the meter perturbs the thing it measures.

### 4.4 Memory as a hard constraint

`program.md:35` currently makes VRAM a *soft* constraint. Make it hard — reject
anything over e.g. 20 GB — and the search redirects toward activation
checkpointing, GQA (`n_kv_head` is currently pinned equal to `n_head` at
`train.py:475` and is an obvious untouched knob), and shorter windows. Also makes
the whole thing runnable on smaller cards, which is what most of the forks in the
README are wrestling with.

---

## 5. Different evaluation metrics

Changing what "good" means, rather than what's held constant.

### 5.1 Inference cost, not just quality

`score = val_bpb × (ms_per_token)^α`, or bpb subject to a decode-latency ceiling.
There is no generation path in the repo at all today — you'd add a KV-cache-free
`benchmark_decode` to `objective.py`. Pushes hard toward GQA, sliding windows
(`WINDOW_PATTERN`, `train.py:435`), and shallow-wide shapes. Probably the single
most *practically* interesting goal swap in this list, because train-time and
inference-time optima genuinely diverge.

### 5.2 Out-of-distribution bpb

The val shard is pinned to one ClimbMix shard (`prepare.py:43`). Score instead on a
different distribution — code, non-English, a much older/newer crawl — or on a
weighted mix. Requires downloading a second val source, which means a new file
rather than a `prepare.py` edit. Turns the loop into a *generalization* search, and
the results might contradict the in-distribution ones in useful ways.

### 5.3 Worst-case rather than average

Score the 90th-percentile per-document bpb instead of the aggregate. `evaluate_bpb`
already computes per-token losses (`prepare.py:359`), so the per-document
aggregation is a small change. Optimizes tail behavior; likely to favor different
architectures than mean loss.

### 5.4 Downstream task instead of likelihood

A tiny synthetic eval the model can plausibly do at this scale — in-context recall
(induction-head style: repeat a random string), a small multiple-choice set scored
by likelihood, or arithmetic. At ~50M params and 5 minutes these will be near-floor,
so pick tasks that show signal early. The interesting outcome is *disagreement*
with bpb: if the ratchet built on task score keeps different changes than the bpb
ratchet, that's a real finding about what bpb misses.

### 5.5 Seed-averaged score

Not a new goal so much as a fix. Run 2-3 seeds at a proportionally shorter budget
and score the mean (or `mean + std`, to reward stability). See §6.1 for why this
matters more than it looks.

---

## 6. Different selection rules (`program.md` changes only)

The cheapest experiments in this whole document — no Python at all.

### 6.1 The current ratchet is statistically fragile

`program.md:103-104` keeps any change that lowers `val_bpb` by any amount. Over
~100 overnight experiments against a **pinned** val shard (`prepare.py:43`), that's
~100 accept/reject decisions made against one fixed sample, with no noise model.
A meaningful fraction of a long run's "improvements" are plausibly seed luck that
got ratcheted in and never revisited. Two fixes, roughly independent:

1. **Require a margin.** Measure the seed-to-seed std of `val_bpb` once (run the
   baseline 5 times unchanged), then only keep changes exceeding ~1σ. This costs
   25 minutes and would inform every subsequent decision.
2. **Hold out a test shard.** Score against val for the ratchet, then confirm the
   final result on a shard the loop never saw. Any gap between the two is a direct
   measurement of how much the run overfit its own selection process.

### 6.2 Pareto ratchet instead of scalar

Keep an archive of non-dominated `(bpb, memory, latency)` points rather than a
single running best. Requires the multi-metric logging from §2.3. Harder for the
agent to reason about, but avoids baking in a weighting you picked arbitrarily.

### 6.3 Constrained ratchet

Simpler and probably better than §6.2 for a first attempt: one primary metric plus
hard constraints ("minimize bpb; reject if VRAM > 20GB or decode > 8 ms/token").
Easy to state in `program.md`, easy for the agent to apply, no scalarization
weights to defend.

### 6.4 Exploration quotas

`program.md` currently says nothing about the *distribution* of ideas, and a greedy
ratchet plus a helpful agent tends toward small safe hyperparameter nudges. Try:
"every 5th experiment must be a structural change, not a hyperparameter change,"
or "maintain a list of 10 untried ideas and refresh it every 20 experiments."
Purely a prompt change, immediately testable, and directly in the spirit of the
repo's framing that `program.md` is the thing the human iterates on.

### 6.5 Parallel goals across branches

The branch naming at `program.md:92` already anticipates multiple GPUs
(`autoresearch/mar5-gpu0`). Run different `GOAL` values on different branches from
the same baseline, then diff what each kept. The comparison between ratchets is
arguably more informative than any single ratchet's output.

---

## 7. What to try first

Ranked by (value ÷ effort), assuming one overnight run each.

| # | Change | Effort | Risk | Why |
|---|---|---|---|---|
| 1 | Seed-noise measurement + keep-margin (§6.1) | 30 min | none | Everything else's results are uninterpretable without it |
| 2 | Split `forward` from `training_loss` (§1.1) | 30 min | none | Unblocks all of §3; fixes a live footgun |
| 3 | Exploration quota in `program.md` (§6.4) | 10 min | low | Zero code, tests the repo's own central claim |
| 4 | Fixed token budget (§4.1) | 1 hr | low | Changes the character of the search the most per line changed |
| 5 | Multi-token prediction (§3.1) | 2 hrs | med | Well-precedented, eval stays comparable |
| 6 | `objective.py` scaffolding (§2.1) | 3 hrs | low | Makes goals a config value; needed for §4-5 |
| 7 | Latency-aware score (§5.1) | 4 hrs | med | Needs a decode path from scratch, but most practically useful |
| 8 | Sequence-length curriculum (§3.3) | 2 hrs | med | Cheap, no `prepare.py` edit, plausible win under a time budget |
| 9 | OOD val set (§5.2) | 4 hrs | med | New data plumbing; findings may not be actionable |
| 10 | Pareto ratchet (§6.2) | 4 hrs | high | Do §6.3 first and see if you actually need this |

---

## 8. Failure modes to expect

Each new goal comes with a way to satisfy it while learning nothing:

- **Throughput or memory goals** → the agent shrinks the model toward zero. Always
  pair a resource goal with a quality floor.
- **Any metric computed in `train.py`** → gets optimized directly rather than
  through the model. Keep the judge in a read-only file, always.
- **Token/FLOP budgets** → wall clock becomes unbounded and the overnight loop
  completes 8 experiments instead of 100. Cap it.
- **Time-to-target** → the agent tunes the schedule to spike past the threshold and
  then diverge. Score the confirmed eval, not the mid-training probe.
- **Downstream task scores** → near-floor at this scale, so the ratchet ends up
  amplifying noise. Check the metric has dynamic range on the baseline *before*
  making it the goal.
- **Multi-objective scalarization** → the weights, not the search, determine the
  answer. Report the frontier alongside whatever scalar you picked.

---

## References

- [A Guide to Andrej Karpathy's AutoResearch: Automating ML with AI Agents](https://www.datacamp.com/tutorial/guide-to-autoresearch) — DataCamp
- `README.md`, `program.md` in this repo
- [nanochat](https://github.com/karpathy/nanochat) — the parent repo this was simplified from
