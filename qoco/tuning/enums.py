from __future__ import annotations

from enum import Enum, auto


class Direction(Enum):
    MINIMIZE = auto()
    MAXIMIZE = auto()

    @property
    def optuna_value(self) -> str:
        return "minimize" if self is Direction.MINIMIZE else "maximize"


class BackendChoice(Enum):
    AER = auto()
    FAKE_KYIV = auto()
    CUSTOM = auto()  # use user-provided backend instance


class InitialStateChoice(Enum):
    NONE = auto()
    UNIFORM_H = auto()


class MixerChoice(Enum):
    NONE = auto()
    X_MIXER = auto()
    Y_MIXER = auto()
    XY_GROUP_RING = auto()
    XY_GROUP_COMPLETE = auto()


class AggregationChoice(Enum):
    NONE = auto()
    CVAR = auto()


class InitialPointChoice(Enum):
    NONE = auto()
    CONSTANT = auto()
    FOURIER = auto()


class CallbackChoice(Enum):
    NONE = auto()


class PostProcChoice(Enum):
    NOOP = auto()
    TAKE_BEST = auto()
    MOST_FREQUENT = auto()
    LOCAL_SEARCH = auto()
    PCE = auto()

