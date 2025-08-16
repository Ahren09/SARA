"""SARA's Mistral variant: a projector maps retrieval/compressor embeddings into the
LM hidden space, replacing ``<COMPRESS>`` token positions in the input.

This is a modernized restoration of the canonical SARA implementation. It targets the
installed Transformers version and fixes three correctness issues from the historical code:

1. The historical per-batch placement check ``(input_ids == compress_id).nonzero(as_tuple=True)[0]``
   returned *row* indices for a 2-D ``input_ids`` and asserted they were non-decreasing — which is
   trivially true and validates nothing. It is removed. We instead validate the compress-token count
   **per example** and rely on PyTorch's row-major (example-major, then position-order) boolean-mask
   scatter, which is correct even when ``<COMPRESS>`` positions are not contiguous within an example.

2. The historical check only compared the *global* compress-token / embedding counts. We additionally
   accept and verify **per-example** counts so a misaligned flatten raises instead of silently
   scattering the wrong embedding into the wrong example.

3. ``generate`` is routed through ``inputs_embeds`` for **both** the plain and the retrieval-augmented
   paths, so the returned sequence contains only newly generated tokens on this Transformers version
   (verified: ``generate(inputs_embeds=...)`` returns shape ``(B, max_new_tokens)``). Use
   :func:`extract_generated_text` for decoding so callers never guess whether the prompt is included.

The flat ``retrieval_embeds`` passed to the model MUST be ordered **example-major, then
document-order within each example** — i.e. all of example 0's compressed-document embeddings (in the
order their ``<COMPRESS>`` tokens appear), then example 1's, and so on. This matches the row-major
scatter into the boolean mask below.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import MistralConfig, MistralForCausalLM

from src.const import COMPRESS


class XMistralConfig(MistralConfig):
    model_type = "mistral"

    def __init__(self, projector_type: str = "mlp2x_gelu", retriever_hidden_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size


class Projector(nn.Module):
    """Maps compressor embeddings (``retriever_hidden_size``) into the LM hidden size."""

    def __init__(self, config: XMistralConfig):
        super().__init__()
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", config.projector_type)
        if not mlp_gelu_match:
            raise ValueError(f"Unsupported projector_type: {config.projector_type!r}")
        mlp_depth = int(mlp_gelu_match.group(1))
        modules: list[nn.Module] = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.projector = nn.Sequential(*modules)
        # Cached for shape validation; inferred from config, never hardcoded.
        self.in_features = config.retriever_hidden_size
        self.out_features = config.hidden_size

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        return self.projector(context_embedding)


class XMistralForCausalLM(MistralForCausalLM):
    """Mistral CausalLM that injects projected retrieval embeddings at ``<COMPRESS>`` positions.

    Backwards compatible with a plain Mistral checkpoint: when ``retrieval_embeds`` is never passed
    and there are no ``<COMPRESS>`` tokens, it behaves exactly like ``MistralForCausalLM``.
    """

    config_class = XMistralConfig

    def __init__(self, config: XMistralConfig):
        super().__init__(config)
        self._compress_token_id: Optional[int] = None
        if getattr(config, "retriever_hidden_size", 0) and config.retriever_hidden_size > 0:
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()

    # ---- compress token id -------------------------------------------------
    @property
    def compress_token_id(self) -> int:
        if self._compress_token_id is None:
            raise RuntimeError(
                "compress_token_id is not set. Call `model.compress_token_id = "
                "tokenizer.convert_tokens_to_ids(COMPRESS)` after loading."
            )
        return self._compress_token_id

    @compress_token_id.setter
    def compress_token_id(self, token_id: int) -> None:
        if not isinstance(token_id, int):
            raise ValueError("compress_token_id must be an integer.")
        self._compress_token_id = token_id

    # ---- embedding injection ----------------------------------------------
    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        retrieval_embeds: Optional[torch.Tensor] = None,
        per_example_counts: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Embed ``input_ids`` and overwrite ``<COMPRESS>`` positions with projected embeds.

        ``retrieval_embeds``: flat ``(total_compress, retriever_hidden_size)`` tensor, ordered
        example-major then document-order (see module docstring).
        ``per_example_counts``: optional per-row expected ``<COMPRESS>`` counts; when provided, each
        row's actual count is validated against it so a misaligned flatten raises early.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        if retrieval_embeds is None:
            return inputs_embeds

        if not hasattr(self, "projector"):
            raise RuntimeError(
                "retrieval_embeds was provided but this model has no projector "
                "(retriever_hidden_size <= 0). It was likely loaded as a plain Mistral."
            )

        compress_mask = input_ids == self.compress_token_id  # (B, S) bool, row-major scatter order

        # ---- per-example validation (replaces the meaningless legacy nonzero[0] check) ----
        per_row_counts = compress_mask.sum(dim=1)  # (B,)
        if per_example_counts is not None:
            expected = torch.as_tensor(per_example_counts, device=per_row_counts.device,
                                       dtype=per_row_counts.dtype)
            if expected.shape != per_row_counts.shape or not torch.equal(per_row_counts, expected):
                raise ValueError(
                    f"Per-example <COMPRESS> count mismatch: rows have {per_row_counts.tolist()} "
                    f"compress tokens but caller expected {expected.tolist()}."
                )
        total_compress = int(per_row_counts.sum().item())
        if total_compress != retrieval_embeds.shape[0]:
            raise ValueError(
                f"<COMPRESS> count ({total_compress}) != retrieval_embeds rows "
                f"({retrieval_embeds.shape[0]}). Embeds must be example-major, document-order."
            )

        # ---- shape / dtype / device checks ----
        if retrieval_embeds.shape[-1] != self.projector.in_features:
            raise ValueError(
                f"retrieval_embeds dim {retrieval_embeds.shape[-1]} != projector in_features "
                f"{self.projector.in_features} (config.retriever_hidden_size)."
            )
        retrieval_embeds = retrieval_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        projected = self.projector(retrieval_embeds)
        if projected.shape[-1] != inputs_embeds.shape[-1]:
            raise ValueError(
                f"projector output dim {projected.shape[-1]} != LM hidden size {inputs_embeds.shape[-1]}."
            )

        # Row-major boolean-mask scatter == example-major, position-order. Works for non-contiguous
        # compress positions; correctness depends only on the example-major/doc-order flatten contract.
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[compress_mask] = projected
        return inputs_embeds

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        per_example_counts: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        # Generation after the first step feeds inputs_embeds (step 0) or input_ids+cache (later steps).
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_beginning_of_generation = inputs_embeds is not None
        if at_beginning_of_generation:
            assert not self.training
            assert retrieval_embeds is None

        if not at_beginning_of_generation:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds, per_example_counts)
            input_ids = None
            # During cached incremental decoding, inputs_embeds holds only the new token while
            # attention_mask spans past+current, so only validate alignment at prefill (no cache).
            no_cache = kwargs.get("past_key_values") is None
            if no_cache and attention_mask is not None and inputs_embeds is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (
                    inputs_embeds.shape, attention_mask.shape)
            if self.training and retrieval_embeds is None:
                inputs_embeds = inputs_embeds.requires_grad_()

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    # ---- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        per_example_counts: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Generate from ``inputs_embeds`` for both paths so the returned tensor contains ONLY new tokens.

        On the installed Transformers version, ``generate(inputs_embeds=...)`` returns shape
        ``(B, max_new_tokens)``. Decode the full returned tensor (see :func:`extract_generated_text`).
        """
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported; pass input_ids (+retrieval_embeds).")
        inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds, per_example_counts)
        if attention_mask is not None:
            assert inputs_embeds.shape[1] == attention_mask.shape[1], (
                inputs_embeds.shape, attention_mask.shape)
        return super().generate(attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


def extract_generated_text(tokenizer, generated_ids: torch.Tensor, **decode_kwargs) -> List[str]:
    """Decode generated token IDs to text.

    :meth:`XMistralForCausalLM.generate` always generates from ``inputs_embeds``, so ``generated_ids``
    already contains only the newly generated continuation (no prompt to strip). This is the single
    shared continuation extractor; do not slice by prompt length elsewhere.
    """
    decode_kwargs.setdefault("skip_special_tokens", True)
    return tokenizer.batch_decode(generated_ids, **decode_kwargs)
