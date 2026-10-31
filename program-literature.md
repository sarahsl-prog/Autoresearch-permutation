# autoresearch — literature-guided

A variant of `program.md` in which experiment ideas come from published research,
not only from what you already know. It adapts the search procedure of
*ScholarEvolve* (Yang et al., arXiv:2609.40169) to this repo: split `train.py` into
modules, find papers for each module, group the papers into distinct mechanisms,
implement one mechanism per module against a fixed champion, combine the winners,
and promote a combination only if it clears a noise margin.

**Read `program.md` first. Everything in it still applies** — setup, the read-only
files, the harness contract, `GPT.forward` vs `GPT.training_loss`, the env-var
rules, output format, logging, snapshots, timeouts, crashes, the simplicity
criterion and NEVER STOP. This file replaces only its "The experiment loop"
section, and adds a research step in front of it.

## Knobs (the human edits these)

| Knob | Default | Meaning |
|---|---|---|
| `MUTATIONS_PER_MODULE` | 2 | Papers implemented per module per generation |
| `CROSSOVER_EVALS` | 3 | Combinations actually run per generation |
| `MARGIN` | 0.005 | Minimum score improvement that counts as real. **Placeholder** — replace with the seed σ from `testing-plans/experiment-catalogue.md` A1 |
| `STALL_GENERATIONS` | 2 | Generations without a promotion before a literature refresh |
| `REPAIR_BUDGET` | 2 | Fix attempts for a crashing candidate before it is dropped |

At the defaults a generation is about 10 mutation runs + 3 crossover runs + 1
confirmation run, so roughly 75 minutes of GPU time, plus your reading time.

## Why this differs from the plain ratchet

The plain loop commits a change, keeps it if the score went down by any amount,
and builds the next idea on top. Two problems with that, both from
`alternative-goals.md` §6: ideas drift toward small hyperparameter nudges from your
own memory, and any lucky draw below the running best gets ratcheted in for good.
Here, every candidate in a generation is measured against **the same champion
commit**, so gains are comparable and can be added up; and nothing is promoted
without clearing `MARGIN` twice.

## The five modules

Each mutation touches exactly one module. The module boundaries are also how
combinations get assembled, so respect them: a change that has to touch two
modules belongs to the module where its main idea lives, and you say so in the
blueprint.

| Module | Where in `train.py` | Examples of mechanisms |
|---|---|---|
| **arch** | `GPTConfig`, attention, `MLP`, `Block`, `GPT.forward`; `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN` | attention variants, gating, normalisation placement, embedding tricks, depth/width |
| **objective** | `GPT.training_loss` only | multi-token prediction, z-loss, label smoothing, loss weighting |
| **optim** | `MuonAdamW` and its fused steps; `*_LR`, `WEIGHT_DECAY`, `ADAM_BETAS` | orthogonalised updates, second-moment variants, per-group LRs, weight-decay forms |
| **schedule** | `get_lr_multiplier`, `get_muon_momentum`, `get_weight_decay`; `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE` | WSD schedules, batch-size ramps, momentum warmup |
| **systems** | compile settings, dtype handling, attention backend selection, data/compute overlap | anything that buys more tokens inside the fixed 5 minutes without changing the maths |

Note that **systems** wins pay out through the other modules: in a fixed wall-clock
budget, 10% more tokens is a quality improvement. The judge measures MFU and wall
clock itself, so there is nothing to gain from mis-reporting them.

## Working files

Keep research notes in `research/<tag>/` (gitignored, so `git reset --hard` on the
ratchet branch never touches them):

```
research/<tag>/
  gaps.md              # current capability gaps (step R1)
  pool-<module>.md     # retrieved papers per module (R2)
  topics-<module>.md   # mechanism topics + assignments (R3)
  screened.md          # every paper ever considered, with verdict — never re-screen
  bp-g<N>-<module>-<k>.md   # one blueprint per mutation (R4)
  generations.md       # per-generation table of gains and the promotion decision
```

Commit only `train.py` on the ratchet branch (`git add train.py`, never `git add -A`).

When you snapshot `results.jsonl` to `main` (see `program.md`), also copy the notes:
`cp -r research/<tag> results/research-<tag>-$(date +%m%d%Y)` and add that
directory in the same commit. Same additive-only rule as the results snapshots.

## Setup additions

After the setup in `program.md`, and after the baseline run:

1. Tag the baseline commit as the first champion: `git tag champ/<tag>/g0`.
2. Write down its score in `research/<tag>/generations.md`. This is `S_champ`.
3. Check your research access. Try your web-search or fetch tool on
   `https://export.arxiv.org/api/query?search_query=all:muon+optimizer&max_results=3`.
   If you have no tool and `curl` works, use `curl` on that API (it returns Atom
   XML with titles and abstracts) and on `https://arxiv.org/abs/<id>` for full pages.
   If neither works, you are in **offline mode**: say so in `generations.md`, and in
   R2 draw papers from what you remember and from the references in
   `testing-plans/` and `alternative-goals.md`. Mark every such entry
   `source: recall` and never invent arXiv IDs, authors or numbers — a recalled
   mechanism with an honest label is fine; a fabricated citation is not.

## The research step (once per cycle, refreshed on stalls)

**R1. Gap audit.** Read the champion's `run.log`, its `run-*.json`, and
`results.jsonl` (every past attempt, including crashes). Turn what you see into
3–8 *capability gaps*, each one sentence, each naming the module it points at.
Examples of the kind of evidence to look for:

- MFU well below what this GPU should reach → **systems**
- loss still falling steeply at the end of the budget → **schedule** / **systems**
- large VRAM headroom → **arch** (bigger model) or **schedule** (bigger batch)
- a family of attempts in `results.jsonl` that all crashed or all regressed → a gap
  in how that idea was implemented, not necessarily in the idea

Gaps are statements about this setup, not about any benchmark leaderboard.

**R2. Retrieve.** For each module, write 2–4 search queries from its gaps plus one
broad query for the module as a whole (e.g. "optimizer for small transformer
pretraining"). Collect 10–20 candidate papers per module, title + abstract + ID,
into `pool-<module>.md`. Drop any paper already in `screened.md`.

**R3. Group into topics.** For each module, read the pool and induce a short list
of *mechanism topics*: a name and a one-line description of the intervention.
Merge paraphrases into one topic; absorb over-specific variants into the broader
mechanism; delete vague topics. Give each paper one primary topic. Then screen the
topics, and drop any that:

- is already in `train.py` (Muon, RoPE, value embeddings, QK-norm and friends —
  read the code, don't assume),
- needs a package that isn't in `pyproject.toml`,
- needs more than one GPU, a pretrained teacher you don't have, or a different
  eval,
- has already been tried here, per `results.jsonl`, unless you have a specific
  reason the earlier attempt was implemented wrong.

Order the surviving topics by cluster size and pick `MUTATIONS_PER_MODULE` papers
**round-robin across topics** — one from the largest topic, one from the next, and
so on — never two from the same topic while another topic is untouched. This is
the whole point: the budget goes to *different mechanisms*, not ten versions of
the one you'd have picked anyway. Record every paper and its verdict in
`screened.md`.

**R4. Blueprint.** For each selected paper, read the method section (and the
appendix and code, if available), then write `bp-g<N>-<module>-<k>.md`:

```
paper:        <title> (<arXiv ID or "source: recall">)
topic:        <mechanism topic>
module:       <module>
mechanism:    what it does, in two or three sentences
assumptions:  what the paper's setting has that ours might not
              (scale, tokens, batch size, sequence length, hardware)
change:       the exact edit to train.py, at function/constant level
hyperparams:  values from the paper, and how you scaled them to ~11M params / 5 min
cost:         expected effect on VRAM and tokens/s
expectation:  predicted score change and the most likely way it fails here
```

The `assumptions` line matters most. Most published tricks are tuned for runs a
thousand times larger than five minutes on one GPU; say which way that should cut
before you see the number.

## The loop

LOOP FOREVER, one generation at a time. `C` is the current champion commit,
`S_champ` its score, `N` the generation number.

**1. Mutate.** For each module, for each of its blueprints:

```
git checkout autoresearch/<tag> && git reset --hard C
# implement the blueprint in train.py — this module only
uv run python harness_check.py
git add train.py && git commit -m "g<N>/<module>: <mechanism> (<paper ID>)"
AUTORESEARCH_NOTE="g<N>/<module>/<k>: <mechanism> [<paper ID>]" \
AUTORESEARCH_TESTING_PLAN="program-literature.md" \
AUTORESEARCH_AGENT="<agent>" AUTORESEARCH_MODEL="<model>" \
timeout 900 uv run train.py > run.log 2>&1
grep "^goal:\|^score:\|^val_bpb:\|^peak_vram_mb:" run.log
git tag cand/<tag>/g<N>-<module>-<k>
uv run python -c "import objective; objective.record_decision('discard')"
```

Every mutation run is recorded `discard`: it did not advance the branch. Whether it
becomes part of a champion is decided in step 3 and lives in `generations.md`. The
tag keeps the commit reachable after the reset.

If it crashes, fix it in place up to `REPAIR_BUDGET` times — an adaptation error at
the boundary between the paper and this code is expected, and is not evidence
against the mechanism. Past the budget, `log_crash`, note it in `screened.md` with
the reason, and move on.

The gain of each mutation is `g = S_champ − score` (positive is better, since the
score is minimised). Write every `g` to `generations.md`.

**2. Combine.** Take the best mutation from each module, if its `g ≥ MARGIN`. If
fewer than two modules qualify, skip to step 3. Otherwise rank every combination of
two or more of those per-module bests by **predicted gain** — the sum of their
individual `g`. Always include the full combination of all of them. Run the top
`CROSSOVER_EVALS`:

```
git checkout autoresearch/<tag> && git reset --hard C
git cherry-pick cand/<tag>/g<N>-arch-1 cand/<tag>/g<N>-optim-2 ...
```

Cherry-picks from different modules can conflict on adjacent lines of the
hyperparameter block. Resolve those by hand — they are textual, not semantic. If two
changes genuinely fight (e.g. both rewrite the LR schedule), that combination is
invalid; skip it. Run, tag `cand/<tag>/g<N>-x<k>`, record `discard`, as in step 1.

Write each combination's **actual** gain next to its predicted gain. The gap
between them is the most informative number in this whole procedure: it tells you
which modules interact.

**3. Select.** Among all candidates of this generation (single mutations and
combinations), take the one with the largest actual gain. Apply the simplicity
criterion from `program.md` — a combination that adds 60 lines for 0.0005 more than
a single mutation loses to the single mutation.

If the best gain is `≥ MARGIN`, **confirm** it: reset to that candidate's commit and
run it once more with the same command (`AUTORESEARCH_NOTE="g<N>/confirm: ..."`). The
seed is fixed, so this measures run-to-run nondeterminism rather than seed luck,
but it still catches flukes. Take the **worse** of the two scores. If that still
clears `MARGIN`:

```
uv run python -c "import objective; objective.record_decision('keep')"
git checkout autoresearch/<tag> && git reset --hard <candidate commit>
git tag champ/<tag>/g<N>
```

The branch now points at the new champion. Update `C` and `S_champ`. If the
confirmation run fails the margin, record it `discard` and leave the champion
unchanged.

**4. Next generation.**

- If the champion changed, start generation `N+1` from step 1 with the next
  unselected papers in each module's topics (continuing round-robin). Rerun the R1
  gap audit first — a new champion has new gaps.
- If `STALL_GENERATIONS` generations in a row produced no promotion, the current
  pool is exhausted for this champion. Do a **literature refresh**: rerun R1–R3 with
  new queries — a different angle on each gap, more recent papers, adjacent fields
  (efficient-training work, small-model scaling studies). Skip everything in
  `screened.md`. This is the "lifelong" part: new reading, not new random nudges.
- If a module has run out of surviving topics, give its budget to the module whose
  mutations had the largest positive gains this cycle.

Never stop on a stall. A stall means read more, not that the work is done.

## What to report in the morning

`generations.md` should let the human answer these without opening `results.jsonl`:

- each generation's champion score, and what got promoted (paper, module, gain),
- predicted versus actual gain for every combination,
- the papers that looked promising and lost, with your best guess at why
  (usually the `assumptions` line),
- whether you ran in offline mode for any of it.
