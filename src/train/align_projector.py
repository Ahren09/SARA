"""Stage B: projector-alignment warm-up (release plan §4), self-contained and pipeline-light.

Trains ONLY the SARA projector (base LM + tokenizer frozen) to reconstruct a document's text from its
single compressed SFR embedding, so the projector learns a useful SFR->Mistral mapping BEFORE the matched
QA fine-tune. Saves the warmed projector via the standard ``save_sara_extras`` so the fine-tune can load it
with ``init_projector_from`` (no change to the QA training/eval pipeline). SARA-specific; RAG has no projector.

Run (GPU 0):
    CUDA_VISIBLE_DEVICES=0 python -m src.train.align_projector \
        --train data/reformatted/QASPER_paragraphs_train.jsonl \
        --out checkpoints/paragraph/sara_qasper_align --steps 600 --batch-size 8 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.const import COMPRESS
from src.eval.compressor_models import CompressorFactory
from src.model.loader import PUBLIC_BASE, build_model_and_tokenizer, save_sara_extras
from src.prompts import MISTRAL_CHAT_TEMPLATE

PROMPT = "Reproduce the following document, which has been compressed into one token: {c}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compressor", default="Salesforce/SFR-Embedding-Mistral")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"

    rhs = 4096
    model, tok, info = build_model_and_tokenizer(
        PUBLIC_BASE, rhs, dtype=torch.bfloat16, attn_implementation="sdpa",
        chat_template=MISTRAL_CHAT_TEMPLATE, hf_token=os.environ.get("HF_TOKEN"))
    model = model.to(dev)
    # Freeze everything except the projector.
    for p in model.parameters():
        p.requires_grad_(False)
    model.projector.requires_grad_(True)
    model.train()
    compressor = CompressorFactory.create_compressor(args.compressor, dev, dtype=torch.bfloat16)

    texts = [json.loads(l)["text"] for l in open(args.train)]
    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.projector.parameters(), lr=args.lr)
    cid = tok.convert_tokens_to_ids(COMPRESS)

    def build_batch(docs):
        input_ids, labels = [], []
        for doc in docs:
            prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(c=COMPRESS)}],
                                             tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            d_ids = tok(doc + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p_ids + d_ids)[: args.max_len]
            lab = ([-100] * len(p_ids) + d_ids)[: args.max_len]
            input_ids.append(ids)
            labels.append(lab)
        maxlen = max(len(x) for x in input_ids)
        pad = tok.pad_token_id
        attn = [[1] * len(x) + [0] * (maxlen - len(x)) for x in input_ids]
        input_ids = [x + [pad] * (maxlen - len(x)) for x in input_ids]
        labels = [x + [-100] * (maxlen - len(x)) for x in labels]
        return (torch.tensor(input_ids, device=dev), torch.tensor(attn, device=dev),
                torch.tensor(labels, device=dev))

    step, t0 = 0, time.perf_counter()
    done = False
    while not done:
        for docs in loader:
            embeds = compressor.get_embeddings(list(docs)).to(dev)  # (B, 4096), one per doc
            input_ids, attn, labels = build_batch(list(docs))
            counts = [1] * len(docs)
            out = model(input_ids=input_ids, attention_mask=attn, retrieval_embeds=embeds,
                        per_example_counts=counts, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                gnorm = max((p.grad.abs().max().item() for p in model.projector.parameters()
                             if p.grad is not None), default=0.0)
                print(f"[align] step {step}/{args.steps} loss {out.loss.item():.4f} "
                      f"proj_grad_max {gnorm:.2e} {step/(time.perf_counter()-t0):.2f} it/s", flush=True)
            if step >= args.steps:
                done = True
                break

    os.makedirs(args.out, exist_ok=True)
    tok.save_pretrained(args.out)
    save_sara_extras(model, info["added_token_ids"], args.out, rhs)
    with open(os.path.join(args.out, "align_meta.json"), "w") as f:
        json.dump({"steps": step, "lr": args.lr, "batch_size": args.batch_size,
                   "n_docs": len(texts), "objective": "paragraph_reconstruction",
                   "trainable": "projector_only", "base": PUBLIC_BASE}, f, indent=2)
    print(f"[align] DONE: saved warmed projector to {args.out}", flush=True)


if __name__ == "__main__":
    main()
