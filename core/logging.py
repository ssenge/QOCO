from __future__ import annotations

import json
import logging
import os
import platform
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if is_dataclass(x):
        return _jsonable(asdict(x))
    name = getattr(x, "__class__", type(x)).__name__
    return {"__type__": name, "__repr__": repr(x)}


def new_run_id(*, prefix: str | None = None) -> str:
    t = time.strftime("%Y%m%d-%H%M%S")
    s = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}-{t}-{s}"
    return f"{t}-{s}"


def default_output_dir() -> Path:
    return Path(os.getenv("QOCO_OUTPUT_DIR", "outputs"))


def default_runs_dir() -> Path:
    return default_output_dir() / "runs"


_LOGGING_RUN_DIR: Path | None = None
_QOCO_LOGGER_NAME = "qoco"


def configure_run_logging(run_dir: Path) -> logging.Logger:
    """Configure stdout + file logging for the current run."""
    global _LOGGING_RUN_DIR
    logger = logging.getLogger(_QOCO_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if _LOGGING_RUN_DIR == run_dir and logger.handlers:
        return logger

    # Remove existing handlers we previously attached.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_path = Path(run_dir) / "run.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    _LOGGING_RUN_DIR = Path(run_dir)
    return logger


class RunLogger:
    def log(self, event: str, payload: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, *, step: int, metrics: Mapping[str, float], prefix: str | None = None) -> None:
        p = {str(k if prefix is None else f"{prefix}{k}"): float(v) for k, v in metrics.items()}
        self.log("step", {"step": int(step), "metrics": p})

    def close(self) -> None:
        return None


class NullLogger(RunLogger):
    def log(self, event: str, payload: Mapping[str, Any]) -> None:
        return None


class JsonlLogger(RunLogger):
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fp = (self.run_dir / "events.jsonl").open("a", encoding="utf-8")

    def log(self, event: str, payload: Mapping[str, Any]) -> None:
        rec = {
            "ts": time.time(),
            "event": str(event),
            "payload": _jsonable(dict(payload)),
        }
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fp.flush()

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        (self.run_dir / "meta.json").write_text(json.dumps(_jsonable(dict(meta)), indent=2), encoding="utf-8")

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def start_run(
    *,
    name: str,
    kind: str,
    config: Mapping[str, Any],
    runs_dir: Path | None = None,
    run_id: str | None = None,
) -> JsonlLogger:
    rid = run_id or new_run_id(prefix=kind)
    root = runs_dir or default_runs_dir()
    run_dir = Path(root) / rid
    configure_run_logging(run_dir)
    logger = JsonlLogger(run_dir)
    logger.write_meta(
        {
            "run_id": rid,
            "name": str(name),
            "kind": str(kind),
            "cwd": str(Path.cwd()),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "executable": os.getenv("PYTHON", None) or os.sys.executable,
            "env": {"CONDA_DEFAULT_ENV": os.getenv("CONDA_DEFAULT_ENV")},
            "config": dict(config),
        }
    )
    logger.log("run_start", {"run_id": rid, "name": name, "kind": kind, "config": dict(config)})
    return logger

