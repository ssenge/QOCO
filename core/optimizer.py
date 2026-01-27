"""Abstract optimizer interface.

Optimizers use a Converter to transform a problem, then optimize it.
Includes always-on structured run logging.
"""

import atexit
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

from .converter import Converter
from qoco.converters.identity import IdentityConverter
from .logging import JsonlLogger, start_run

# Type variables
P = TypeVar("P")  # Problem type
T = TypeVar("T")  # Converted problem type
R = TypeVar("R")  # Result type


@dataclass
class Optimizer(ABC, Generic[P, T, R]):
    """
    Abstract optimizer that converts a problem and optimizes it.

    Type parameters:
        P: Problem type (must extend Problem)
        T: Converted problem type
        R: Result type
    """

    converter: Converter[P, T] = field(default_factory=IdentityConverter)
    _logger: JsonlLogger | None = field(default=None, init=False, repr=False)
    _logger_started: float | None = field(default=None, init=False, repr=False)

    def optimize(self, problem: P, *, log: bool = True) -> R:
        """Convert problem and optimize, with timing.

        Args:
            log: If True (default), write structured run logs. If False, do not start/log a run.
        """
        if log and self._logger is None:
            try:
                self._logger = start_run(
                    name=self.__class__.__name__,
                    kind="opt",
                    config={
                        "optimizer": self.__class__.__name__,
                        "converter": self.converter.__class__.__name__,
                    },
                )
                self._logger_started = time.time()
                atexit.register(self.close)
            except Exception:
                self._logger = None
                self._logger_started = None

        start = time.perf_counter()
        converted = self.converter.convert(problem)
        result = self._optimize(converted)
        elapsed = time.perf_counter() - start

        try:
            out = replace(result, tts=elapsed)
        except Exception:
            out = result

        if log and self._logger is not None:
            metrics: dict[str, float] = {"opt/tts_s": float(elapsed)}
            status = getattr(out, "status", None)
            objective = getattr(out, "objective", None)
            if objective is not None:
                metrics["opt/objective"] = float(objective)
            info = getattr(out, "info", None)
            if isinstance(info, dict) and "infer_s" in info:
                try:
                    metrics["opt/infer_s"] = float(info["infer_s"])
                except Exception:
                    pass
            payload = {
                "problem": getattr(problem, "name", problem.__class__.__name__),
                "status": str(getattr(status, "name", status)) if status is not None else None,
                "metrics": metrics,
            }
            self._logger.log("optimize", payload)

        return out

    def close(self) -> None:
        if self._logger is None:
            return
        payload: dict[str, float | str] = {}
        if self._logger_started is not None:
            payload["wall_s"] = float(time.time() - self._logger_started)
        self._logger.log("run_end", payload)
        self._logger.close()
        self._logger = None
        self._logger_started = None

    @abstractmethod
    def _optimize(self, converted: T) -> R:
        """Optimize the converted problem. Implement in subclass."""
        raise NotImplementedError

