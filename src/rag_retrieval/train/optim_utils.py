"""Shared optimizer helpers for rag_retrieval training scripts."""

import torch


def create_adamw_optimizer(
    model,
    lr,
    weight_decay=1e-2,
    no_decay_keywords=('bias', 'LayerNorm', 'layernorm'),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            'weight_decay': 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
