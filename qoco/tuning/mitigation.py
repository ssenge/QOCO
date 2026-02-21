from __future__ import annotations

from typing import Any

from qoco.tuning.enums import MitigationProfile


def apply_mitigation_profile(*, sampler: Any, profile: MitigationProfile) -> None:
    if profile == MitigationProfile.NONE:
        return
    if profile == MitigationProfile.BASIC_DD_XY4_TWIRLING_AUTO:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"
        return
    raise ValueError(f"Unknown mitigation profile: {profile}")

