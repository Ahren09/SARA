"""Shared helpers for ``src/run_sara.py`` and ``src/run_baselines.py``.

These two top-level entry points replace the 27 bash scripts that used to live
in ``scripts/``. Anything that both runners need (matrix iteration, output-path
templating, subprocess wrapping, accelerate launcher construction) lives here.
"""

from __future__ import annotations

import itertools
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_kn_pairs(values: Sequence[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw in values:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"--kn expects 'K,N' (got {raw!r}); pass it once per pair, e.g. --kn 5,5 --kn 5,10"
            )
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def parse_int_csv(value: str) -> list[int]:
    return [int(v) for v in parse_csv(value)]


def make_output_path(
    *,
    model: str,
    dataset: str,
    retriever: str | None = None,
    k: int | None = None,
    n: int | None = None,
    num_evidence: int | None = None,
    num_additional_evidence: int | None = None,
    target_token: int | None = None,
    repetition_penalty: float | None = None,
    evidence_selection: str | None = None,
    base_dir: str = "outputs",
) -> Path:
    """Build the canonical eval output path.

    Mirrors the existing bash naming so downstream
    ``src/eval/calculate_*_performance.py`` scripts keep finding old files:

      outputs/{model}/answers_{model}[_{retriever}]_{dataset}
                     [_k{k}_n{n}|_num-evi{ne}_num-add{na}|_num-evi{ne}]
                     [_target{T}][_rep{R}][_evi-select-{evi}].jsonl
    """
    parts = [f"answers_{model}"]
    if retriever:
        parts.append(Path(retriever).name)
    parts.append(dataset)
    if k is not None and n is not None:
        parts.append(f"k{k}")
        parts.append(f"n{n}")
    elif num_evidence is not None and num_additional_evidence is not None:
        parts.append(f"num-evi{num_evidence}")
        parts.append(f"num-add{num_additional_evidence}")
    elif num_evidence is not None:
        parts.append(f"num-evi{num_evidence}")
    if target_token is not None:
        parts.append(f"target{target_token}")
    if repetition_penalty is not None:
        parts.append(f"rep{repetition_penalty}")
    if evidence_selection is not None:
        parts.append(f"evi-select-{evidence_selection}")
    return Path(base_dir) / model / ("_".join(parts) + ".jsonl")


def iter_matrix(**axes: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """Cartesian product over named axes, preserving insertion order."""
    keys = list(axes.keys())
    values = [list(axes[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def should_skip(output_path: Path, skip_existing: bool) -> bool:
    return skip_existing and output_path.exists()


@dataclass
class SubprocessResult:
    cmd: list[str]
    returncode: int
    skipped: bool = False


def run_subprocess(
    cmd: Sequence[str],
    *,
    gpu: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    dry_run: bool = False,
    label: str | None = None,
) -> SubprocessResult:
    """Run ``cmd`` with ``CUDA_VISIBLE_DEVICES`` set, mirroring the bash style.

    Prints the resolved command before executing (the equivalent of ``set -x``).
    With ``dry_run=True`` only prints, returns ``returncode=0`` without spawning.
    """
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if extra_env:
        env.update(extra_env)

    rendered = " ".join(shlex.quote(c) for c in cmd)
    prefix = f"[{label}] " if label else ""
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    cvd_str = f"CUDA_VISIBLE_DEVICES={cvd} " if cvd is not None else ""
    print(f"{prefix}+ {cvd_str}{rendered}", flush=True)

    if dry_run:
        return SubprocessResult(cmd=list(cmd), returncode=0)

    completed = subprocess.run(list(cmd), env=env, check=False)
    return SubprocessResult(cmd=list(cmd), returncode=completed.returncode)


@dataclass
class AccelerateOptions:
    num_processes: int = 1
    num_machines: int = 1
    mixed_precision: str = "bf16"
    main_process_port: int = 29666

    def to_args(self) -> list[str]:
        return [
            "accelerate",
            "launch",
            "--mixed_precision",
            self.mixed_precision,
            "--num_machines",
            str(self.num_machines),
            "--num_processes",
            str(self.num_processes),
            "--main_process_port",
            str(self.main_process_port),
        ]


def build_module_cmd(
    module: str,
    args: Sequence[str],
    *,
    accelerate: AccelerateOptions | None = None,
) -> list[str]:
    """Build a ``python -m <module>`` or ``accelerate launch -m <module>`` cmd."""
    if accelerate is None:
        return [sys.executable, "-m", module, *args]
    return [*accelerate.to_args(), "-m", module, *args]


@dataclass
class SweepReport:
    total: int = 0
    skipped: int = 0
    failed: list[tuple[str, int]] = field(default_factory=list)
    succeeded: int = 0

    def summarize(self) -> str:
        lines = [
            f"Sweep finished: {self.total} combinations, "
            f"{self.succeeded} succeeded, {self.skipped} skipped, {len(self.failed)} failed."
        ]
        for label, rc in self.failed:
            lines.append(f"  FAILED rc={rc}: {label}")
        return "\n".join(lines)


def run_sweep(
    jobs: Iterable[dict[str, Any]],
    *,
    build_cmd: "callable[[dict[str, Any]], tuple[list[str], Path | None, str]]",
    gpu: str | None,
    dry_run: bool,
    skip_existing: bool,
    keep_going: bool,
) -> SweepReport:
    """Iterate ``jobs`` and run each one as a subprocess.

    ``build_cmd(job)`` must return ``(cmd, output_path_or_None, label)``.
    ``output_path`` enables skip-if-exists behavior; pass ``None`` to disable.
    """
    report = SweepReport()
    for job in jobs:
        report.total += 1
        cmd, output_path, label = build_cmd(job)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if should_skip(output_path, skip_existing):
                print(
                    f"[{label}] skip: output exists at {output_path}", flush=True
                )
                report.skipped += 1
                continue

        result = run_subprocess(
            cmd, gpu=gpu, dry_run=dry_run, label=label
        )
        if result.returncode != 0:
            report.failed.append((label, result.returncode))
            if not keep_going:
                print(report.summarize(), flush=True)
                raise SystemExit(result.returncode)
        else:
            report.succeeded += 1

    print(report.summarize(), flush=True)
    return report
