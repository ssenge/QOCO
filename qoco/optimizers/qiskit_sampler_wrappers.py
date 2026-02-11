from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def normalize_measurements(circuit: Any) -> Any:
    """Return a copy with exactly one measurement layer.

    Some Qiskit backend translation paths (notably Braket) reject circuits that
    measure a qubit more than once. This strips existing `measure` ops and appends
    a single `measure_all()`.
    """
    c = circuit.copy()
    c.data = [inst for inst in c.data if inst.operation.name != "measure"]
    c.measure_all()
    return c


@dataclass
class MeasurementNormalizedSamplerV2:
    """SamplerV2 wrapper that normalizes circuit measurement structure."""

    base_sampler: Any

    def run(self, pubs: Any, *, shots: int | None = None) -> Any:
        from qiskit.primitives.containers.sampler_pub import SamplerPub

        coerced = [SamplerPub.coerce(pub) for pub in pubs]
        normalized = [
            SamplerPub(
                normalize_measurements(pub.circuit),
                pub.parameter_values,
                pub.shots,
                validate=False,
            )
            for pub in coerced
        ]
        return self.base_sampler.run(normalized, shots=shots)

