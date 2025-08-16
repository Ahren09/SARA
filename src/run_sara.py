"""Single Python entry point for all SARA training and evaluation runs.

Replaces the bash scripts under ``scripts/`` for SARA-method work
(``pretrain``, ``paragraph``, ``finetune``, ``train-reranker``, ``eval``).

The public release supports exactly two evaluated methods — Standard RAG and SARA —
both restricted to physical GPUs 0 and 1.

Usage examples
--------------
    # Standard RAG: retrieved docs inserted into the Mistral prompt as plain text (k == n)
    CUDA_VISIBLE_DEVICES=0,1 python -m src.run_sara evaluate --method rag \\
        --models Mistral7B_QA_Compress_Stage0 --retrievers bm25 --datasets qasper --n 10

    # SARA: k natural-language pieces + (n-k) compressed retrieval embeddings
    CUDA_VISIBLE_DEVICES=0,1 python -m src.run_sara evaluate --method sara \\
        --models Mistral7B_QA_Compress_Stage0 --retrievers bm25 --datasets qasper \\
        --k 5 --n 10 --repetition-penalty 1.5

    # training (two-process, GPUs 0,1 only)
    CUDA_VISIBLE_DEVICES=0,1 python -m src.run_sara finetune \\
        --config config/language_modeling/Finetune_Stage0_Mistral7B.yaml \\
        --launcher accelerate --gpus 0,1 --num-processes 2

A sweep is a cartesian product over the comma-separated lists and repeated ``--kn``
flags. Skip-if-output-exists is on by default so re-running a partial sweep only fills
in the missing combinations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.utils.run_utils import (
    AccelerateOptions,
    build_module_cmd,
    iter_matrix,
    make_output_path,
    parse_csv,
    parse_kn_pairs,
    run_subprocess,
    run_sweep,
)

EVAL_MODULE = "src.eval.eval_single_round"
TRAIN_MODULE = "src.train.train_generator"
RERANKER_MODULE = "src.train.train_reranker"

DEFAULT_EVAL_CONFIG = "config/language_modeling/eval.yaml"
# GPUs are restricted to physical devices 0 and 1 for all SARA work (release plan §7).
DEFAULT_GPU = "0,1"
SUPPORTED_METHODS = ("rag", "sara")


def _add_common_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", default=DEFAULT_GPU,
                   help="Single GPU index for CUDA_VISIBLE_DEVICES (default: 2).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without spawning subprocesses.")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Re-run combinations even if their output file already exists.")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue the sweep on subprocess failure (default: stop on first failure).")
    p.set_defaults(skip_existing=True)


def _add_accelerate_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--launcher", choices=["python", "accelerate"], default="python",
                   help="Use plain python (default) or `accelerate launch`.")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated CUDA devices for multi-GPU launches "
                        "(overrides --gpu when set).")
    p.add_argument("--num-processes", type=int, default=1,
                   help="--num_processes for accelerate launch.")
    p.add_argument("--num-machines", type=int, default=1)
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--main-process-port", type=int, default=29666)


def _build_train_cmd(module: str, config: str, extra_args: list[str],
                     accel: AccelerateOptions | None) -> list[str]:
    args = ["--config", config, *extra_args]
    return build_module_cmd(module, args, accelerate=accel)


def _resolve_gpu(args: argparse.Namespace) -> str:
    if getattr(args, "gpus", None):
        return args.gpus
    return args.gpu


def _maybe_accelerate(args: argparse.Namespace) -> AccelerateOptions | None:
    if getattr(args, "launcher", "python") != "accelerate":
        return None
    return AccelerateOptions(
        num_processes=args.num_processes,
        num_machines=args.num_machines,
        mixed_precision=args.mixed_precision,
        main_process_port=args.main_process_port,
    )


def cmd_train_stage(args: argparse.Namespace, *, module: str, label: str) -> int:
    """Single-run training stage: forward --config and any passthrough args.

    Anything after ``--`` on the command line is forwarded verbatim to the
    underlying training entry point (handy for e.g. ``-- --learning_rate 1e-5``).
    """
    accel = _maybe_accelerate(args)
    extra: list[str] = list(getattr(args, "passthrough", None) or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd = _build_train_cmd(module, args.config, extra, accel)
    result = run_subprocess(
        cmd, gpu=_resolve_gpu(args), dry_run=args.dry_run, label=label,
    )
    return result.returncode


def _resolve_method_kn(method: str, k: int, n: int) -> tuple[int, int]:
    """Map a method to the (k, n) evidence contract. rag => text-only (k==n); sara => k<n."""
    if method not in SUPPORTED_METHODS:
        raise SystemExit(f"Unsupported --method {method!r}. Choose one of {SUPPORTED_METHODS}.")
    if method == "rag":
        # Standard RAG: all n retrieved candidates rendered as text, no compression.
        return n, n
    # SARA: k natural-language pieces + (n-k) compressed; require at least one compressed piece.
    if not (0 <= k < n):
        raise SystemExit(f"SARA requires 0 <= k < n (got k={k}, n={n}).")
    return k, n


def cmd_evaluate(args: argparse.Namespace) -> int:
    method = args.method
    if method not in SUPPORTED_METHODS:
        raise SystemExit(f"Unsupported --method {method!r}. Choose one of {SUPPORTED_METHODS}.")
    if method == "rag" and args.evidence_selections:
        raise SystemExit("--evidence-selections is a SARA-only option; omit it for --method rag.")

    models = parse_csv(args.models)
    retrievers = parse_csv(args.retrievers) if args.retrievers else [None]
    datasets = parse_csv(args.datasets)
    raw_kn = parse_kn_pairs(args.kn) if args.kn else [(args.k, args.n)]
    kn_pairs = [_resolve_method_kn(method, k, n) for (k, n) in raw_kn]
    evi_selects: list[str | None]
    if args.evidence_selections:
        evi_selects = parse_csv(args.evidence_selections)
    else:
        evi_selects = [None]

    if not models or not datasets:
        raise SystemExit("--models and --datasets are required for evaluate.")

    jobs = iter_matrix(
        model=models,
        retriever=retrievers,
        dataset=datasets,
        kn=kn_pairs,
        evi_select=evi_selects,
    )

    config = args.config or DEFAULT_EVAL_CONFIG
    explicit_model_config: str | None = args.model_config
    explicit_output_file: str | None = args.output_file

    def build(job: dict[str, Any]) -> tuple[list[str], Path | None, str]:
        model: str = job["model"]
        retriever: str | None = job["retriever"]
        dataset: str = job["dataset"]
        k, n = job["kn"]
        evi_select: str | None = job["evi_select"]

        model_config = explicit_model_config or f"config/model/{model}.yaml"
        dataset_config = f"config/dataset/{dataset}.yaml"
        output_path = (
            Path(explicit_output_file)
            if explicit_output_file
            else make_output_path(
                model=model,
                retriever=retriever,
                dataset=dataset,
                k=k,
                n=n,
                repetition_penalty=args.repetition_penalty,
                evidence_selection=evi_select,
            )
        )

        cli: list[str] = [
            "--dataset_config", dataset_config,
            "--config", config,
            "--output_file", str(output_path),
            "--model_config", model_config,
            "--method", method,
            "--k", str(k),
            "--n", str(n),
        ]
        if args.repetition_penalty is not None:
            cli += ["--repetition_penalty", str(args.repetition_penalty)]
        if retriever is not None:
            cli += ["--retriever_name_or_path", retriever]
        if evi_select is not None:
            cli += ["--evidence_selection", evi_select]
        if args.max_eval_samples is not None:
            cli += ["--max_eval_samples", str(args.max_eval_samples)]
        if args.batch_size is not None:
            cli += ["--batch_size", str(args.batch_size)]

        cmd = build_module_cmd(EVAL_MODULE, cli)
        label = f"evaluate[{method}] {model} | {retriever or '-'} | {dataset} | k={k} n={n} evi={evi_select or '-'}"
        return cmd, output_path, label

    report = run_sweep(
        jobs,
        build_cmd=build,
        gpu=_resolve_gpu(args),
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        keep_going=args.keep_going,
    )
    return 0 if not report.failed else 1


def cmd_report(args: argparse.Namespace) -> int:
    from src.eval.calculate_qa_performance import build_comparison_report
    build_comparison_report(args.rag, args.sara, args.out_dir, exp_id=args.exp_id,
                            target_improvement=args.target)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_sara.py",
        description="Unified runner for SARA training and evaluation. "
                    "Replaces the bash scripts under scripts/.",
    )
    sub = p.add_subparsers(dest="stage", required=True)

    # ---- training stages (single run each) ----
    for name, default_launcher in (
        ("pretrain", "accelerate"),
        ("paragraph", "python"),
        ("finetune", "python"),
    ):
        sp = sub.add_parser(name, help=f"Run the {name} stage of SARA training.")
        sp.add_argument("--config", required=True,
                        help=f"YAML under config/language_modeling/ for the {name} stage.")
        _add_accelerate_flags(sp)
        # Override default launcher per stage.
        sp.set_defaults(launcher=default_launcher)
        _add_common_run_flags(sp)
        sp.add_argument("passthrough", nargs=argparse.REMAINDER,
                        help="Args after `--` are forwarded verbatim to the training entry point.")

    # ---- reranker training ----
    sp = sub.add_parser("train-reranker", help="Train the cross-encoder reranker (accelerate).")
    sp.add_argument("--config", required=True,
                    help="YAML config (e.g. config/language_modeling/pretrain.yaml).")
    _add_accelerate_flags(sp)
    sp.set_defaults(launcher="accelerate")
    _add_common_run_flags(sp)
    sp.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="Args after `--` are forwarded verbatim to src.train.train_reranker.")

    # ---- evaluate (Standard RAG or SARA) ----
    sp = sub.add_parser("evaluate", help="Evaluate Standard RAG or SARA (--method).")
    sp.add_argument("--method", required=True, choices=list(SUPPORTED_METHODS),
                    help="rag = retrieved docs as plain text (k==n); sara = k text + (n-k) compressed.")
    sp.add_argument("--models", required=True,
                    help="Comma-separated list of model names. Each maps to "
                         "config/model/{name}.yaml unless --model-config overrides.")
    sp.add_argument("--model-config", default=None,
                    help="Override the model_config YAML for all models in the sweep.")
    sp.add_argument("--retrievers", default="",
                    help="Comma-separated retrievers (e.g. bm25,BAAI/bge-reranker-v2-m3). "
                         "Empty to skip the flag entirely.")
    sp.add_argument("--datasets", required=True,
                    help="Comma-separated dataset names. Each maps to config/dataset/{name}.yaml.")
    sp.add_argument("--kn", action="append", default=None,
                    help="Repeated 'K,N' pair (e.g. --kn 5,5 --kn 5,10). "
                         "Use --k/--n for a single combination instead.")
    sp.add_argument("--k", type=int, default=5,
                    help="Number of evidence pieces in natural language format (default: 5). "
                         "Ignored if --kn is given.")
    sp.add_argument("--n", type=int, default=10,
                    help="Total number of evidence pieces (default: 10). Ignored if --kn is given.")
    sp.add_argument("--evidence-selections", default=None,
                    help="Comma-separated evidence-selection methods "
                         "(self-info,embed,kl-div). Empty to skip the flag.")
    sp.add_argument("--repetition-penalty", type=float, default=None)
    sp.add_argument("--max-eval-samples", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None,
                    help="Per-device eval batch size for batched generation.")
    sp.add_argument("--config", default=None,
                    help=f"Top-level eval config YAML (default: {DEFAULT_EVAL_CONFIG}).")
    sp.add_argument("--output-file", default=None,
                    help="Override the output path. Only useful for single-combination runs.")
    _add_common_run_flags(sp)

    # ---- comparison report ----
    sp = sub.add_parser("report", help="Build the Standard RAG vs SARA comparison report + table.")
    sp.add_argument("--rag", required=True, help="Standard RAG predictions JSONL.")
    sp.add_argument("--sara", required=True, help="SARA predictions JSONL.")
    sp.add_argument("--out-dir", default="outputs/docs/sara_vs_rag_qasper")
    sp.add_argument("--exp-id", default="sara_vs_rag_qasper")
    sp.add_argument("--target", type=float, default=5.0)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "evaluate":
        return cmd_evaluate(args)
    if args.stage == "report":
        return cmd_report(args)
    if args.stage == "train-reranker":
        return cmd_train_stage(args, module=RERANKER_MODULE, label="train-reranker")
    if args.stage in ("pretrain", "paragraph", "finetune"):
        return cmd_train_stage(args, module=TRAIN_MODULE, label=args.stage)
    raise SystemExit(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    sys.exit(main())
