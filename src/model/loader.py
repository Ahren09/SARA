"""Single source of truth for building, saving, and loading the SARA Mistral model.

Public-base-first (release plan §4/§8): both Standard RAG and SARA start from a public Mistral base,
add the two SARA tokens deterministically (`<pad>`=32000, `<COMPRESS>`=32001), resize embeddings, and
train LoRA (+ projector for SARA). Checkpoints store ONLY: the LoRA adapter, the projector (via PEFT
``modules_to_save``), the tokenizer, and a small ``added_token_weights.safetensors`` with the embedding /
lm_head rows of the newly added tokens. No merged full checkpoint is ever produced.

Why a dedicated added-token file: `resize_token_embeddings` uses mean-resizing with randomness, so the new
rows are not reproducible across processes; LoRA on q_proj/v_proj does not train base embeddings. Saving the
rows makes a fresh-process reload numerically exact.

Used by both ``src/train/train_generator.py`` and ``src/utils/eval_utils.py`` — do not reimplement loading
elsewhere (CLAUDE rule 4).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from tokenizers import AddedToken
from transformers import AutoTokenizer

from src.const import COMPRESS
from src.model.xMistral import XMistralConfig, XMistralForCausalLM

PUBLIC_BASE = "mistralai/Mistral-7B-Instruct-v0.2"
PAD = "<pad>"
# The projector is persisted explicitly (NOT via PEFT modules_to_save): the Projector.projector nesting
# breaks PEFT's modules_to_save save-map. We store the projector params and the added-token embedding /
# lm_head rows together in one small safetensors file alongside the LoRA adapter.
EXTRAS_FILE = "sara_extras.safetensors"
SARA_CONFIG = "sara_config.json"

# eos-as-pad is intentionally NOT used: training masks labels at pad positions while eos must remain a
# trainable label (src/data/preprocessing.py), so a dedicated <pad> token is required.


def extend_tokenizer(tokenizer, chat_template: Optional[str] = None) -> Tuple[int, int, int]:
    """Add `<pad>` then `<COMPRESS>` deterministically. Returns (num_added, pad_id, compress_id)."""
    num_added = tokenizer.add_special_tokens({"pad_token": PAD})
    assert num_added in (0, 1), "Expected to add at most one pad token."
    num_added += tokenizer.add_tokens([AddedToken(COMPRESS, lstrip=False, rstrip=False)])
    tokenizer.padding_side = "left"  # required for batched left-padded generation from inputs_embeds
    if tokenizer.chat_template is None and chat_template is not None:
        tokenizer.chat_template = chat_template
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    compress_id = tokenizer.convert_tokens_to_ids(COMPRESS)
    return num_added, pad_id, compress_id


def build_model_and_tokenizer(
    base: str,
    retriever_hidden_size: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
    chat_template: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[XMistralForCausalLM, "AutoTokenizer", dict]:
    """Build a fresh XMistral from a (public) base, extend the tokenizer, and resize embeddings.

    ``retriever_hidden_size`` must match the compressor embedding dim (e.g. SFR ``get_embed_dim()``); it
    sizes the projector and is stored in the config. Returns (model, tokenizer, token_info).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        base, use_fast=True, local_files_only=local_files_only, token=hf_token,
    )
    num_added, pad_id, compress_id = extend_tokenizer(tokenizer, chat_template=chat_template)

    config = XMistralConfig.from_pretrained(
        base, retriever_hidden_size=retriever_hidden_size, local_files_only=local_files_only,
        token=hf_token,
    )
    model = XMistralForCausalLM.from_pretrained(
        base, config=config, torch_dtype=dtype, attn_implementation=attn_implementation,
        local_files_only=local_files_only, token=hf_token,
    )
    model.compress_token_id = int(compress_id)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    assert model.config.vocab_size == len(tokenizer), (model.config.vocab_size, len(tokenizer))
    # Standard RAG adapters are trained projector-free (retriever_hidden_size<=0); only assert the
    # projector exists for SARA (retriever_hidden_size>0).
    if retriever_hidden_size > 0:
        _assert_sara_model(model)
    token_info = {
        "base_model": base,
        "added_token_ids": sorted({int(pad_id), int(compress_id)}),
        "pad_id": int(pad_id),
        "compress_id": int(compress_id),
        "final_vocab_size": len(tokenizer),
    }
    return model, tokenizer, token_info


def _assert_sara_model(model) -> None:
    """Fail loudly if a plain (non-SARA) model was loaded for a SARA run."""
    if not isinstance(model, XMistralForCausalLM):
        raise TypeError(f"Expected XMistralForCausalLM, got {type(model).__name__} — SARA needs the projector.")
    if not hasattr(model, "projector"):
        raise RuntimeError("Loaded XMistral has no projector (retriever_hidden_size <= 0).")


def _base_module(model):
    """Return the underlying XMistral whether or not the model is PEFT-wrapped."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def save_sara_extras(model, added_token_ids, output_dir: str, retriever_hidden_size: int) -> None:
    """Persist the projector params and added-token embedding / lm_head rows (LoRA-first, no full ckpt)."""
    base = _base_module(model)
    embed = base.get_input_embeddings().weight.data
    ids = torch.tensor(sorted(int(i) for i in added_token_ids), dtype=torch.long)
    tensors = {"token_ids": ids, "embed_rows": embed[ids].clone().cpu().contiguous()}
    if not base.config.tie_word_embeddings:
        lm_head = base.get_output_embeddings().weight.data
        tensors["lm_head_rows"] = lm_head[ids].clone().cpu().contiguous()
    for n, p in base.projector.state_dict().items():
        tensors[f"projector.{n}"] = p.clone().cpu().contiguous()
    st_save(tensors, os.path.join(output_dir, EXTRAS_FILE))
    with open(os.path.join(output_dir, SARA_CONFIG), "w") as f:
        json.dump({"retriever_hidden_size": int(retriever_hidden_size),
                   "tie_word_embeddings": bool(base.config.tie_word_embeddings),
                   "added_token_ids": [int(i) for i in ids.tolist()]}, f, indent=2)


def load_sara_extras(model, adapter_dir: str) -> None:
    """Restore the projector params and added-token rows in-place from the extras file."""
    path = os.path.join(adapter_dir, EXTRAS_FILE)
    if not os.path.exists(path):
        return
    base = _base_module(model)
    tensors = st_load(path)
    ids = tensors["token_ids"].long()
    embed = base.get_input_embeddings().weight.data
    embed[ids] = tensors["embed_rows"].to(device=embed.device, dtype=embed.dtype)
    if "lm_head_rows" in tensors and not base.config.tie_word_embeddings:
        lm_head = base.get_output_embeddings().weight.data
        lm_head[ids] = tensors["lm_head_rows"].to(device=lm_head.device, dtype=lm_head.dtype)
    proj_sd = {n[len("projector."):]: t for n, t in tensors.items() if n.startswith("projector.")}
    target = base.projector.state_dict()
    proj_sd = {k: v.to(device=target[k].device, dtype=target[k].dtype) for k, v in proj_sd.items()}
    base.projector.load_state_dict(proj_sd)


def load_sara_for_eval(
    adapter_dir: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    chat_template: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[XMistralForCausalLM, "AutoTokenizer"]:
    """Load a SARA (or matched RAG) adapter onto its public base for evaluation (LoRA-first).

    Reads the adapter dir for the base model + projector (PEFT ``modules_to_save``) and restores the
    added-token rows. Asserts the result is an XMistral with a projector. Tokenizer is loaded from the
    adapter dir (it contains the added tokens).
    """
    from peft import PeftConfig, PeftModel

    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    base = peft_cfg.base_model_name_or_path
    # The adapter base path points at the public base; rebuild with the projector dim recorded at train
    # time (sara_config.json) so the projector shape matches before restoring its weights.
    rhs = _read_retriever_hidden_size(adapter_dir, base)

    model, tokenizer, _ = build_model_and_tokenizer(
        base, rhs, dtype=dtype, attn_implementation=attn_implementation,
        chat_template=chat_template, hf_token=hf_token,
    )
    # Prefer the adapter-dir tokenizer (carries added tokens / chat template) when present.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        tokenizer.padding_side = "left"
    except Exception:
        pass
    model = PeftModel.from_pretrained(model, adapter_dir, torch_dtype=dtype)
    load_sara_extras(model, adapter_dir)
    if rhs > 0:  # SARA needs a projector; Standard RAG adapters are projector-free
        _assert_sara_model(_base_module(model))
    _base_module(model).compress_token_id = int(tokenizer.convert_tokens_to_ids(COMPRESS))
    if device != "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _read_retriever_hidden_size(adapter_dir: str, base: str) -> int:
    """Recover the projector input dim recorded at training time (sara_config.json).

    Returns 0 for a projector-free Standard RAG adapter (no sara_config.json) so it loads as a plain
    XMistral with no projector.
    """
    cfg_path = os.path.join(adapter_dir, SARA_CONFIG)
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return int(json.load(f)["retriever_hidden_size"])
    return 0
