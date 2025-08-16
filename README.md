<div align="center">

# 🔍 SARA

### *Adaptive Retrieval & Context Compression for Retrieval-Augmented Generation*

<p>
  <a href="https://ahren09.github.io/">Yiqiao Jin</a><sup>1</sup>,
  <a href="https://ksartik.github.io/">Kartik Sharma</a><sup>1</sup>,
  <a href="https://www.linkedin.com/in/vineethrakesh/">Vineeth Rakesh</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/ytongdou/">Yingtong Dou</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/panmenghai/">Menghai Pan</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/mahashwetadas/">Mahashweta Das</a><sup>2</sup>,
  <a href="https://faculty.cc.gatech.edu/~srijan/">Srijan Kumar</a><sup>1</sup>
</p>

<sub><sup>1</sup> Georgia Institute of Technology &nbsp;·&nbsp; <sup>2</sup> Visa Research</sub>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.57-FFD21E.svg)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/🤗_PEFT-LoRA-FFD21E.svg)](https://github.com/huggingface/peft)
[![Mistral](https://img.shields.io/badge/Base-Mistral_7B-FF7000.svg)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

</div>

---

```bibtex
@inproceedings{jin2025sara,
  title={SARA: Selective and Adaptive Retrieval-augmented Generation with Context Compression},
  author={Jin, Yiqiao and Sharma, Kartik and Rakesh, Vineeth and Dou, Yingtong and Pan, Menghai and Das, Mahashweta and Kumar, Srijan},
  booktitle={ACL},
  year={2026}
}
```

## 📖 Overview

**SARA** represents retrieved evidence as a mix of **`k` natural-language passages** and **`n − k` compressed
retrieval embeddings**. The compressed pieces are injected at `<COMPRESS>` token positions in the prompt
through a small trained **projector** that maps retriever embeddings into the language model's hidden space,
letting the model attend to many more documents within a fixed context budget.

This repository supports **exactly two evaluated methods**, sharing one dataset, retriever, model loader,
generation code, metrics, and CLI:

- **Standard RAG** — all `n` retrieved documents inserted into the Mistral prompt as ordinary text.
- **SARA** — the same `n` candidates, but `k` rendered as text and `n − k` as compressed embeddings.

Both start from the **public** `mistralai/Mistral-7B-Instruct-v0.2` base and are trained with matched budgets,
so the comparison is fair by construction.

## ✨ Highlights

- 🎯 **Two-method, apples-to-apples**: Standard RAG vs SARA under identical data, retriever, prompts, decoding, and seeds.
- 🧠 **Projector + `<COMPRESS>` tokens**: retriever embeddings projected into the LM hidden space at compress positions.
- 🪶 **LoRA-first**: adapters + projector only — no merged or full-model checkpoints are ever saved.
- ⚡ **Fully batched** generation and evaluation (dynamic padding, one GPU call per batch).
- 🔒 **Leakage-safe**: document-level (per-paper) train/dev/test splits.
- 🛠️ **Single CLI**: `python -m src.run_sara {finetune, evaluate, report} --method {rag, sara}`.

## 🚀 Installation

SARA targets **Python 3.11 + CUDA 12.x**. Install PyTorch from the cu128 wheel index first, then the rest.

```bash
# 1) Create and activate the env
conda create -n sara python=3.11 -y
conda activate sara
pip install -U pip setuptools wheel

# 2) Install cu128 PyTorch from the PyTorch wheel index
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# 3) Install SARA + its dependencies (editable)
pip install -e .
#   or:  pip install -r requirements.txt
```

> 💡 **Flash-Attention is optional.** The model loads with PyTorch **SDPA** by default, so no flash-attn build is required.

> 🔑 Set `HF_TOKEN` to access the gated Mistral base. `WANDB_API_KEY` is optional (only if you set `use_wandb: true`).

### Verify the install

```bash
python -c "
import torch, transformers, peft
from src.model import XMistralForCausalLM
print('torch       ', torch.__version__, 'cuda_ok:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('peft        ', peft.__version__)
print('SARA model  ', XMistralForCausalLM.__name__, 'OK')
"
```

## ⚡ Quick Start

All commands run on **GPU 0** (set `CUDA_VISIBLE_DEVICES` to choose devices). QASPER is the primary benchmark.

### 1 — Prepare data (leakage-safe document-level split)

```bash
python -m src.data.make_qasper_splits --dev-frac 0.1 --seed 42
```

### 2 — Train the two adapters from the public Mistral base

```bash
# Standard RAG  (LoRA only; all n documents as text)
CUDA_VISIBLE_DEVICES=0 python -m src.run_sara finetune \
    --config config/language_modeling/Finetune_RAG_qasper_pubbase.yaml

# SARA  (LoRA + projector; k text + (n-k) compressed)
CUDA_VISIBLE_DEVICES=0 python -m src.run_sara finetune \
    --config config/language_modeling/Finetune_SARA_qasper_pubbase.yaml
```

> 🪄 **Optional — projector alignment warm-up.** Pre-train the SARA projector to reconstruct documents from
> their compressed embeddings before fine-tuning (set `init_projector_from` in the SARA config to the output):
> ```bash
> CUDA_VISIBLE_DEVICES=0 python -m src.train.align_projector \
>     --train data/reformatted/QASPER_paragraphs_train.jsonl \
>     --out checkpoints/paragraph/sara_qasper_align --steps 600 --batch-size 8 --lr 5e-4
> ```

### 3 — Evaluate (batched)

```bash
# Standard RAG
CUDA_VISIBLE_DEVICES=0 python -m src.run_sara evaluate --method rag \
    --models rag_qasper --retrievers bm25 --datasets qasper --n 10 --batch-size 16

# SARA
CUDA_VISIBLE_DEVICES=0 python -m src.run_sara evaluate --method sara \
    --models sara_qasper --retrievers bm25 --datasets qasper --k 5 --n 10 --batch-size 16
```

### 4 — Build the comparison report

```bash
python -m src.run_sara report \
    --rag  outputs/rag_qasper/answers_rag_qasper_bm25_qasper_k10_n10_rep1.5.jsonl \
    --sara outputs/sara_qasper/answers_sara_qasper_bm25_qasper_k5_n10_rep1.5.jsonl \
    --out-dir outputs/docs/sara_vs_rag_qasper
```

## 🧩 CLI Reference

| Subcommand | Purpose |
| :--- | :--- |
| `finetune` | Train a LoRA (+ projector for SARA) adapter from the public base. |
| `evaluate` | Run batched evaluation for a method (`--method rag` or `--method sara`). |
| `report` | Build the Standard RAG vs SARA comparison report + table. |

**Evidence allocation** is controlled by `--k` and `--n`:

| Method | Allocation | Flags |
| :--- | :--- | :--- |
| `rag` | all `n` documents as **text** (`k = n`) | `--n N` |
| `sara` | `k` text + `n − k` **compressed** (`k < n`) | `--k K --n N` |

Unsupported `--method` values are rejected with a clear error.

## 📊 Dataset & 🤖 Models

| Component | Default |
| :--- | :--- |
| 📄 **Primary benchmark** | QASPER (long-document scientific QA) |
| 🤖 **Base LM** | `mistralai/Mistral-7B-Instruct-v0.2` |
| 🧬 **Compressor / retriever-embedding** | `Salesforce/SFR-Embedding-Mistral` |
| 🔎 **Retriever** | BM25 |

## 🧪 Tests

```bash
python -m pytest tests/test_xmistral.py \
                 tests/test_checkpoint_roundtrip.py \
                 tests/test_train_sara_sanity.py -q
```

These cover the `<COMPRESS>` embedding injection (batched vs single-example parity), checkpoint round-trip
(LoRA + projector + added-token rows reload numerically), and the training gradient/optimizer sanity checks.

## ⚖️ License

Released under the [Apache License 2.0](LICENSE).

## 🙏 Acknowledgements

Built on excellent open-source work:
[🤗 Transformers](https://github.com/huggingface/transformers) ·
[🤗 PEFT](https://github.com/huggingface/peft) ·
[🤗 Accelerate](https://github.com/huggingface/accelerate) ·
[LlamaIndex](https://github.com/run-llama/llama_index).
