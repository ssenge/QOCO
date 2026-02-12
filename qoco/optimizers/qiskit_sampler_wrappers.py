from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


@dataclass
class CountsBackendSamplerV2:
    """SamplerV2 adapter for backends that don't provide memory.

    Qiskit's `BackendSamplerV2` requires the backend to support `memory=True` and to return
    per-shot memory. Some provider backends (e.g. certain PlanQK/IQM integrations) may return
    counts/probabilities but no memory, which breaks `BackendSamplerV2` postprocessing.

    This adapter executes pubs one-by-one via `backend.run(...)` and synthesizes per-shot
    memory from the returned counts so downstream algorithms can consume SamplerV2 results.
    """

    backend: Any
    default_shots: int = 1024

    def run(self, pubs: Any, *, shots: int | None = None) -> Any:
        from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult
        from qiskit.primitives.containers.sampler_pub import SamplerPub
        from qiskit.primitives.primitive_job import PrimitiveJob

        if shots is None:
            shots = int(self.default_shots)

        coerced = [SamplerPub.coerce(pub, shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs: list[Any]) -> Any:
        from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult

        results: list[Any] = []
        for pub in pubs:
            # Bind parameters (QAOA uses bound circuits); handle potential ndarray output.
            bound = pub.parameter_values.bind_all(pub.circuit)
            circuits = np.ravel(bound).tolist()
            if len(circuits) != 1:
                raise ValueError("CountsBackendSamplerV2 only supports a single circuit per pub")
            circuit = circuits[0]

            # Execute (backend is expected to be BackendV2-like).
            job = self.backend.run(circuit, shots=int(pub.shots))
            res = job.result()

            # Extract counts. Prefer Result.get_counts if available.
            if hasattr(res, "get_counts"):
                counts = res.get_counts()
            else:
                counts = getattr(res, "counts", None)
            if counts is None:
                raise ValueError("backend result has no counts")

            # Synthesize packed memory bytes from counts.
            num_bits = int(getattr(circuit, "num_clbits", 0))
            if num_bits <= 0:
                results.append(SamplerPubResult(DataBin(shape=()), metadata={"shots": int(pub.shots)}))
                continue

            num_bytes = (num_bits + 7) // 8
            rows: list[int] = []
            for bitstring, c in dict(counts).items():
                s = str(bitstring).strip()
                if s.startswith("0b"):
                    s = s[2:]
                s = "".join(ch for ch in s if ch in ("0", "1"))
                if not s:
                    continue
                if len(s) < num_bits:
                    s = s.zfill(num_bits)
                if len(s) > num_bits:
                    s = s[-num_bits:]
                value = int(s, 2)
                rows.extend([value] * int(c))

            # If the backend returned probabilities instead of integer counts, we may be short.
            # In that case, pad with zeros to keep shapes consistent.
            if len(rows) < int(pub.shots):
                rows.extend([0] * (int(pub.shots) - len(rows)))
            if len(rows) > int(pub.shots):
                rows = rows[: int(pub.shots)]

            data = b"".join(int(v).to_bytes(num_bytes, "big") for v in rows)
            packed = np.frombuffer(data, dtype=np.uint8).reshape(int(pub.shots), num_bytes)

            creg_name = circuit.cregs[0].name if getattr(circuit, "cregs", None) else "meas"
            meas = {str(creg_name): BitArray(packed, num_bits)}
            results.append(
                SamplerPubResult(
                    DataBin(**meas, shape=()),
                    metadata={"shots": int(pub.shots), "circuit_metadata": getattr(circuit, "metadata", {})},
                )
            )

        return PrimitiveResult(results, metadata={"version": 2})

