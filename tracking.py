"""
MLflow experiment tracking for autoresearch.

This file belongs to the harness, not to the agent. It is read-only in the same
sense prepare.py is: train.py gets rewritten every experiment, but the tracking
calls inside it must survive, or an overnight run logs nothing.

Three constraints shape the design:

  1. Never crash a training run. Every MLflow failure degrades to a one-line
     warning and the run continues untracked. A dead tracking server must cost
     you telemetry, not a night of experiments.

  2. Never do network I/O inside the timed loop. train.py measures per-step `dt`
     to enforce the time budget and report MFU; a blocking HTTP POST in that
     window would corrupt both. Per-step metrics are buffered in memory (a list
     append) and flushed in batches after training ends.

  3. Never stall. Short HTTP timeouts and few retries, so an unreachable server
     costs seconds per experiment rather than minutes.

Usage from train.py:

    import tracking
    tracking.start(params=tracking.collect_hyperparams(globals()))
    ...
    tracking.log_step(step, {"train_loss": ...})     # buffered, no I/O
    ...
    tracking.log_summary({"val_bpb": ...})
"""

import os
import time
import socket
import atexit
import subprocess

# Defaults live here so train.py needs no configuration. Override either with an
# environment variable.
DEFAULT_TRACKING_URI = "http://192.168.0.252:5000"
DEFAULT_EXPERIMENT = "autoresearch"

# Anchor to this file's directory rather than the cwd, so git metadata and the
# train.py artifact are still captured when a run is launched from elsewhere.
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Must be set before mlflow is imported. MLflow's stock behaviour is 5 retries
# with exponential backoff on a 120s timeout, which turns an unreachable server
# into a multi-minute stall on every single experiment.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

try:
    import mlflow
    from mlflow.entities import Metric, Param

    _MLFLOW_IMPORT_ERROR = None
except Exception as e:  # ImportError, or a broken install
    mlflow = None
    _MLFLOW_IMPORT_ERROR = e

# MLflow's server rejects oversized log_batch calls (1000 entities total,
# 100 params). Stay under both.
_METRIC_CHUNK = 900
_PARAM_CHUNK = 90

_MAX_PARAM_LEN = 500  # MLflow truncates/rejects longer param values


class _State:
    def __init__(self):
        self.enabled = False
        self.run_id = None
        self.client = None
        self.buffer = []  # list[Metric], flushed after training
        self.finished = False


_state = _State()


def _warn(msg):
    print(f"[tracking] {msg}", flush=True)


def _disable(msg):
    """Turn tracking off for the rest of the process. Never raises."""
    if _state.enabled:
        _warn(f"disabling tracking: {msg}")
    _state.enabled = False


def _git(*args, default=""):
    try:
        out = subprocess.run(
            ["git", "-C", _REPO_DIR, *args], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else default
    except Exception:
        return default


def _git_tags():
    dirty = bool(_git("status", "--porcelain"))
    return {
        "git_commit": _git("rev-parse", "--short=7", "HEAD", default="unknown"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", default="unknown"),
        "git_dirty": str(dirty).lower(),
        "hostname": socket.gethostname(),
    }


def _device_tags():
    tags = {}
    try:
        import torch

        tags["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            tags["gpu"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability()
            tags["gpu_capability"] = f"{cap[0]}.{cap[1]}"
    except Exception:
        pass
    return tags


def collect_hyperparams(namespace):
    """
    Pull UPPERCASE module-level constants out of a namespace, e.g.
    tracking.collect_hyperparams(globals()).

    Deliberately name-agnostic: the agent renames, adds and deletes
    hyperparameters constantly, and a hardcoded list would silently go stale.
    Anything ALL_CAPS and scalar-ish gets logged.
    """
    params = {}
    for key, value in namespace.items():
        if not key.isupper() or key.startswith("_"):
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            params[key] = value
        elif isinstance(value, (tuple, list)) and all(
            isinstance(v, (int, float, str, bool)) for v in value
        ):
            params[key] = str(value)
    return params


def _preflight(uri, timeout=2.0):
    """
    Cheap TCP reachability check before handing off to MLflow.

    MLflow's own retry policy turns an unreachable host into ~35s per run, which
    over an overnight loop is about an hour of lost wall clock. A refused or
    timed-out connect here costs 2s instead. Returns True for non-HTTP URIs
    (sqlite, databricks, ...), which have no host to probe.
    """
    try:
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        if parsed.scheme not in ("http", "https"):
            return True
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        socket.create_connection((parsed.hostname, port), timeout=timeout).close()
        return True
    except Exception as e:
        _warn(f"{uri} unreachable ({type(e).__name__}), running untracked")
        return False


def start(params=None, tags=None, note=None, experiment=None, tracking_uri=None):
    """
    Open an MLflow run. Safe to call when MLflow is missing or the server is
    down — tracking just stays disabled.
    """
    if mlflow is None:
        _warn(f"mlflow not importable ({_MLFLOW_IMPORT_ERROR}), running untracked")
        return

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    exp = experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT
    note = note or os.environ.get("AUTORESEARCH_NOTE")

    if not _preflight(uri):
        _state.enabled = False
        return

    all_tags = {}
    all_tags.update(_git_tags())
    all_tags.update(_device_tags())
    if tags:
        all_tags.update(tags)
    if note:
        all_tags["mlflow.note.content"] = note
        all_tags["note"] = note

    run_name = os.environ.get("AUTORESEARCH_RUN_NAME") or (
        f"{all_tags.get('git_commit', 'nogit')}-{time.strftime('%H%M%S')}"
    )

    t0 = time.time()
    try:
        mlflow.set_tracking_uri(uri)
        experiment_id = mlflow.set_experiment(exp).experiment_id
        client = mlflow.tracking.MlflowClient()
        run = client.create_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={k: str(v) for k, v in all_tags.items()},
        )
        _state.client = client
        _state.run_id = run.info.run_id
        _state.enabled = True
    except Exception as e:
        # Unreachable server, auth failure, bad URI — all non-fatal.
        _warn(
            f"could not start run at {uri} ({type(e).__name__}: {e}), running untracked"
        )
        _state.enabled = False
        return

    if params:
        _log_params(params)

    _warn(
        f"logging to {uri} | experiment={exp} | run={run_name} "
        f"({time.time() - t0:.1f}s)"
    )
    atexit.register(_atexit_handler)


def _log_params(params):
    if not _state.enabled:
        return
    entities = []
    for key, value in params.items():
        text = str(value)
        if len(text) > _MAX_PARAM_LEN:
            text = text[:_MAX_PARAM_LEN]
        entities.append(Param(str(key), text))
    try:
        for i in range(0, len(entities), _PARAM_CHUNK):
            _state.client.log_batch(
                _state.run_id, params=entities[i : i + _PARAM_CHUNK]
            )
    except Exception as e:
        _disable(f"log_batch(params) failed ({type(e).__name__}: {e})")


def log_step(step, metrics):
    """
    Buffer per-step metrics. Pure in-memory append — no network, no blocking.
    Call this OUTSIDE the region train.py times for its budget.
    """
    if not _state.enabled:
        return
    now_ms = int(time.time() * 1000)
    for key, value in metrics.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        # NaN/inf are rejected server-side and would fail the whole batch.
        if value != value or value in (float("inf"), float("-inf")):
            continue
        _state.buffer.append(Metric(str(key), value, now_ms, int(step)))


def flush():
    """Send buffered per-step metrics. Called automatically by log_summary."""
    if not _state.enabled or not _state.buffer:
        return
    buffered, _state.buffer = _state.buffer, []
    t0 = time.time()
    try:
        for i in range(0, len(buffered), _METRIC_CHUNK):
            _state.client.log_batch(
                _state.run_id, metrics=buffered[i : i + _METRIC_CHUNK]
            )
        _warn(f"flushed {len(buffered)} step metrics in {time.time() - t0:.1f}s")
    except Exception as e:
        _disable(f"log_batch(metrics) failed ({type(e).__name__}: {e})")


def log_summary(metrics, artifacts=()):
    """Log final run metrics (val_bpb and friends), then flush the step buffer."""
    if not _state.enabled:
        return
    flush()
    if not _state.enabled:
        return
    now_ms = int(time.time() * 1000)
    entities = []
    for key, value in metrics.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value:
            continue
        entities.append(Metric(str(key), value, now_ms, 0))
    try:
        _state.client.log_batch(_state.run_id, metrics=entities)
    except Exception as e:
        _disable(f"log_batch(summary) failed ({type(e).__name__}: {e})")
        return

    # The source of the experiment is the experiment. Log it so a run in the UI
    # is self-describing even after the branch moves on.
    for path in (os.path.join(_REPO_DIR, "train.py"), *artifacts):
        try:
            if os.path.exists(path):
                _state.client.log_artifact(_state.run_id, path)
            else:
                _warn(f"artifact not found, skipping: {path}")
        except Exception as e:
            _warn(f"could not log artifact {path} ({type(e).__name__}: {e})")

    diff = _git("diff", "HEAD~1", "--", "train.py")
    if diff:
        try:
            _state.client.log_text(_state.run_id, diff, "train.py.diff")
        except Exception as e:
            _warn(f"could not log diff ({type(e).__name__}: {e})")


def finish(status="FINISHED"):
    if not _state.enabled or _state.finished:
        return
    flush()
    _state.finished = True
    try:
        _state.client.set_terminated(_state.run_id, status=status)
    except Exception as e:
        _warn(f"could not terminate run ({type(e).__name__}: {e})")


def _atexit_handler():
    """
    train.py can exit(1) on a NaN loss, OOM, or any agent-introduced bug. Without
    this the run sits in RUNNING forever and pollutes the experiment list.
    """
    if _state.enabled and not _state.finished:
        finish(status="FAILED")
