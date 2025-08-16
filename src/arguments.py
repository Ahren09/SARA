import argparse
import os
import sys

from src.utils.utility import get_yaml_file


def set_attributes_from_yaml(args, yaml_config, overwrite=False):
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None or overwrite:
            setattr(args, key, value)
    return args

def get_answer_file_path(output_dir, model_name_or_path, retriever_name_or_path, dataset_name, k, n, repetition_penalty, evidence_selection: str = None, reconstruct_context: bool = None):
    path = f"{output_dir}/{model_name_or_path.split('/')[-1]}/answers_{model_name_or_path.split('/')[-1]}_{retriever_name_or_path.split('/')[-1]}_{dataset_name}_k{k}_n{n}_rep{repetition_penalty}"

    
    if evidence_selection:
        path += f"_evi-select-{evidence_selection}"
        
    if reconstruct_context:
        path += "_reconstruct"
        
    path += ".jsonl"
    
    return path


def parse_args(mode):
    if mode not in ["train", "eval", "data", "metrics", "eval_compress_token"]:
        raise ValueError("Mode must be 'train', 'eval', 'data', 'metrics', or 'eval_compress_token'")

    parser = argparse.ArgumentParser()

    # Shared arguments (sorted alphabetically)
    parser.add_argument("--cache_dir", type=str, default='cache')
    parser.add_argument("--chat_format", choices=['mistral', 'tulu', 'mixtral', 'qwen', 'yi', 'gemma', 'llama'], default='mistral')
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--compressor_name_or_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embedding_model", type=str, default=None)
    parser.add_argument("--enable_fsdp", action='store_true')
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--exclude_dataset_type", nargs="+", default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_note", type=str, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1000_000_000)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qa_model", type=str, default=None)
    parser.add_argument("--quantization", type=str)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--reranker_name_or_path", type=str, default="/workingdir/yjin328/SARA/checkpoints/reranker/mlp_epoch3.pth")
    parser.add_argument("--retriever_name_or_path", type=str, default=None, choices=["bm25", "BAAI/bge-reranker-v2-m3", "Salesforce/SFR-Embedding-Mistral"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_sentences", action="store_true")
    parser.add_argument("--summarization_model", type=str, default=None)
    parser.add_argument("--target_token", type=int, default=None)
    parser.add_argument("--task_type", type=str, choices=["pretrain", "paragraph", "finetune", "finetune_stage1", "metrics", "data"], default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_fast_tokenizer", type=eval)
    parser.add_argument("--k", type=int, default=5, help="Number of evidence pieces in natural language format")
    parser.add_argument("--n", type=int, default=10, help="Total number of evidence pieces (natural language + compressed tokens)")
    parser.add_argument("--method", choices=["rag", "sara"], default=None,
                        help="Evaluated method: rag = retrieved docs as plain text; sara = k text + (n-k) compressed.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device eval batch size for batched generation (overrides per_device_eval_batch_size).")

    if mode == "train":
        parser.add_argument("--base_model", help='base LLM load')
        parser.add_argument("--distill_topk", type=int)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--index_start", type=int, default=0)
        parser.add_argument("--learning_rate", type=float)
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--output_model_name", type=str, default=None)
        parser.add_argument("--per_device_train_batch_size", type=int, default=None)
        parser.add_argument("--save_on_epoch_end", action='store_true')
        parser.add_argument("--use_rag_tuning", type=eval)
        parser.add_argument("--warmup_proportion", type=float, default=0.05)
        parser.add_argument("--workdir", type=str)

    elif mode in {"eval", "eval_compress_token"}:
        parser.add_argument("--enable_progress_bar", type=eval, default=True)
        parser.add_argument("--evidence_selection", type=str, choices=["self-info", "embed", "kl-div"], default=None)
        parser.add_argument("--model_config", type=str, required=True)
        parser.add_argument("--num_compressed_sentences", type=int, default=None)
        parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
        parser.add_argument("--sample_dataset", action='store_true')
        parser.add_argument("--split", type=str, default=None, choices=["train", "validation", "test"])
        parser.add_argument("--tf_idf_topk", type=int, default=0)
        parser.add_argument("--topk", type=int, default=3)
        parser.add_argument("--use_rag", action='store_true')

    elif mode == "data":
        parser.add_argument("--topk", type=int, default=3)

    if mode in ["eval", "eval_compress_token", "metrics"]:
        parser.add_argument("--output_file", type=str)
        parser.add_argument("--dataset_config", type=str, required=False)
        parser.add_argument("--track_metrics", action='store_true')

    args = parser.parse_args()

    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    args.data_dir = os.path.abspath(os.path.expanduser(args.data_dir))

    for path_attr in ["qa_model", "summarization_model"]:
        path = getattr(args, path_attr, None)
        if isinstance(path, str) and path.startswith("checkpoint"):
            setattr(args, path_attr, os.path.abspath(os.path.join(args.output_dir, "..", path)))

    assert os.path.exists(args.config), f"Experiment config {args.config} does not exist."
    args = set_attributes_from_yaml(args, get_yaml_file(args.config))

    if not hasattr(args, "use_trainer"):
        args.use_trainer = False

    if args.task_type == "finetune_stage1":
        assert args.k == 1

    if args.task_type in ["eval", "eval_compress_token"] and mode != "data":
        model_config = get_yaml_file(args.model_config)
        args = set_attributes_from_yaml(args, model_config)

    if args.task_type in ["eval", "metrics", "data"] and args.dataset_config:
        dataset_config = get_yaml_file(args.dataset_config)
        args = set_attributes_from_yaml(args, dataset_config)

    if mode in ["train", "eval", "eval_compress_token"]:
        for feature in ["model_name_or_path", "num_workers", "chunk_size"]:
            assert getattr(args, feature, None) is not None

    if mode in ["eval", "eval_compress_token"]:
        assert args.repetition_penalty is not None
        if not hasattr(args, "reconstruct_context"):
            args.reconstruct_context = False

    if args.debug:
        args.num_workers = 1

    args.use_hf_pipeline = getattr(args, "use_hf_pipeline", False)

    if mode == "train":
        os.makedirs(args.output_dir, exist_ok=True)

    if getattr(args, "output_file", None):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print(args)
    return args
