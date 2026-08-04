"""
harness_check.py — static guard on train.py's contract with the harness.

train.py is rewritten by the agent every experiment. Most breakage is loud: if a
change stops the model training the run crashes, and the ratchet discards it.
Three kinds are silent, and this file exists for those:

  1. **A dropped `tracking.*` call.** The run trains, scores, and gets ratcheted
     normally. It just never appears in MLflow, and nothing in the output says so.

  2. **`tracking.log_step` moved inside the timed window.** The run trains and
     scores, and reports a `training_seconds` and MFU that quietly include
     network latency — corrupting the budget the whole experiment rests on.

  3. **A loss in `GPT.forward`.** `evaluate_bpb` calls it, so a training
     objective placed there changes the number that decides whether that same
     experiment gets kept.

Stdlib only, and it *parses* rather than imports — no torch, no GPU, no data.
Two entry points, because the two places this can break don't overlap:

  - `objective.py` calls `check()` at import, so a broken edit shows up at the
    top of run.log inside the research loop, where CI never runs.
  - CI runs `python harness_check.py` as a hard gate on pushed branches.

Checks are conservative on purpose. A false failure here would train the agent to
work around the guard, which is worse than the breakage it prevents — so anything
that can't be established with confidence is reported as "skipped", not failed.
"""

import ast
import os
import sys

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGET = os.path.join(_REPO_DIR, "train.py")

# Call -> why it has to survive a rewrite.
REQUIRED_CALLS = {
    "objective.report": "scores the run, prints the summary, writes results.jsonl",
    "objective.log_crash": "records diverged runs so the denominator stays honest",
    "tracking.start": "opens the MLflow run",
    "tracking.log_step": "buffers the training curve",
    "tracking.log_summary": "sends the final metrics",
    "tracking.finish": "closes the run so it isn't left RUNNING",
}


def _dotted(node):
    """'objective.report' for an Attribute/Name chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _called_names(tree):
    """Map of dotted call name -> list of line numbers."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name:
                found.setdefault(name, []).append(node.lineno)
    return found


def _find_training_loop(tree):
    """The while/for whose body contains a .backward() call."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.While, ast.For)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                if inner.func.attr == "backward":
                    return node
    return None


def _method(tree, class_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return item
    return None


def check(path=DEFAULT_TARGET):
    """
    Returns (problems, skipped) — both lists of human-readable strings.
    An empty `problems` means the contract holds as far as this can tell.
    """
    problems, skipped = [], []

    try:
        source = open(path, encoding="utf-8").read()
    except OSError as e:
        return [f"could not read {path}: {e}"], []

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        return [
            f"{os.path.basename(path)} does not parse: line {e.lineno}: {e.msg}"
        ], []

    calls = _called_names(tree)

    # 1. Every harness call still present.
    for name, why in REQUIRED_CALLS.items():
        if name not in calls:
            problems.append(f"missing call to {name}() — {why}")

    # 2. log_step outside the timed window. The window ends at the last
    #    time.time() read in the loop; anything after that is free.
    loop = _find_training_loop(tree)
    log_step_lines = calls.get("tracking.log_step", [])
    if not log_step_lines:
        pass  # already reported above
    elif loop is None:
        skipped.append(
            "could not locate the training loop; log_step placement unchecked"
        )
    else:
        timing_lines = [
            n.lineno
            for n in ast.walk(loop)
            if isinstance(n, ast.Call) and _dotted(n.func) == "time.time"
        ]
        in_loop = [
            ln for ln in log_step_lines if loop.lineno <= ln <= (loop.end_lineno or ln)
        ]
        if not timing_lines:
            skipped.append(
                "no time.time() in the training loop; log_step placement unchecked"
            )
        elif not in_loop:
            skipped.append(
                "tracking.log_step is not inside the training loop; placement unchecked"
            )
        elif min(in_loop) < max(timing_lines):
            problems.append(
                f"tracking.log_step (line {min(in_loop)}) runs before the last "
                f"time.time() in the training loop (line {max(timing_lines)}). It must "
                f"sit outside the timed window, or network latency is charged to the "
                f"time budget and MFU."
            )

    # 3. The eval contract: forward() still computes plain cross-entropy, and
    #    training_loss() still exists to hold anything that isn't that.
    forward = _method(tree, "GPT", "forward")
    training_loss = _method(tree, "GPT", "training_loss")
    if forward is None:
        skipped.append("no GPT.forward found; eval contract unchecked")
    else:
        names = _called_names(forward)
        if not any(n.endswith("cross_entropy") for n in names):
            problems.append(
                "GPT.forward no longer calls cross_entropy. evaluate_bpb calls "
                "forward() and treats the result as plain next-token CE — putting a "
                "different loss there silently changes the score that decides "
                "whether this experiment is kept. Use GPT.training_loss instead."
            )
    if training_loss is None:
        problems.append(
            "GPT.training_loss is gone. It is where training objectives belong; "
            "without it they end up in forward(), which is the eval contract."
        )

    return problems, skipped


def warn(path=DEFAULT_TARGET, prefix="[harness]"):
    """Print problems without raising. Used at objective.py import time."""
    try:
        problems, _ = check(path)
    except Exception as e:  # never let the guard break a training run
        print(f"{prefix} guard failed to run ({type(e).__name__}: {e})")
        return []
    for problem in problems:
        print(f"{prefix} WARNING: {problem}")
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else DEFAULT_TARGET
    problems, skipped = check(path)

    for note in skipped:
        print(f"skipped: {note}")
    if problems:
        print(f"\n{os.path.basename(path)} breaks its contract with the harness:\n")
        for problem in problems:
            print(f"  - {problem}")
        print(f"\n{len(problems)} problem(s). See program.md.")
        return 1
    print(f"{os.path.basename(path)}: harness contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
