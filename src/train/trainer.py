from __future__ import annotations

import os
from typing import Any, Sized

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class Trainer:
    def __init__(
            self,
            *,
            model: torch.nn.Module,
            train_dataloader: DataLoader,
            optimizer: Optimizer,
            accelerator: Accelerator,
            validation_dataloader: DataLoader | None = None,
            epochs: int = 3,
            lr_scheduler: LRScheduler | None = None,
            log_interval: int = 50,
            save_on_epoch_end: bool = True,
            tokenizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end
        self.tokenizer = tokenizer

        self.validation_loss_tracker = LossTracker()
        if isinstance(self.train_dataloader.dataset, Sized):
            num_steps_per_epoch = len(self.train_dataloader)
        else:
            num_steps_per_epoch = None

        self.current_step = 0

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to WandB via the accelerator."""
        self.accelerator.log(metrics, step=step)

    def train(self):
        for current_epoch in range(1, self.epochs + 1):
            self.model.train()

            for batch_index, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {current_epoch}")):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()

                    batch_output = self.model(**batch)

                    loss = batch_output['loss']

                    self.accelerator.backward(loss)
                    # for name, param in self.model.mlp.named_parameters():
                    #     print(name, param.grad)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                self.current_step += 1
                if (batch_index + 1) % self.log_interval == 0:
                    print(
                        f"Epoch={current_epoch}\tbatch={batch_index}\tloss: {loss.item()}, lr: {self.lr_scheduler.get_lr()[0]}")
                    self.log_metrics(
                        {
                            'loss': loss.item(),
                            'lr': float(self.lr_scheduler.get_lr()[0])
                        },
                        step=self.current_step,
                    )

                if (batch_index + 1) % 2000 == 0 and self.validation_dataloader:
                    ret = evaluate(
                        self.model,
                        self.validation_dataloader,
                        self.validation_loss_tracker,
                    )

                    # Log validation metrics
                    self.log_metrics(
                        self.add_prefix(ret, 'validation'),
                        step=self.current_step,
                    )

                    validation_loss = ret['loss']
                    # validation_metrics = self.add_prefix({'loss': validation_loss}, 'validation')
                    self.accelerator.print(f'Epoch {current_epoch}\tBatch {batch_index}\tVal loss: {validation_loss:.4f}')
                    for name, value in ret.items():
                        self.accelerator.print(f'{name}: {value:.4f}\t')
                    

            if self.save_on_epoch_end:
                if self.accelerator.is_local_main_process:
                    # save_dir=self.get_checkpoint_dir(current_epoch)
                    save_dir = os.path.join(self.accelerator.project_dir, "checkpoints", 'reranker')
                    os.makedirs(save_dir, exist_ok=True)

                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.mlp.state_dict(),
                               os.path.join(save_dir, f'mlp_epoch{current_epoch}.pth'))
                self.accelerator.wait_for_everyone()

        self.accelerator.end_training()

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}


def evaluate(
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_tracker: LossTracker | None = None,
):
    model = model.eval()
    loss_tracker = loss_tracker or LossTracker()
    predicted_labels, true_labels = [], []
    for batch in dataloader:
        with torch.inference_mode():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            batch_output = model(**batch)
            loss_tracker.update(batch_output['loss'])
            predicted_labels.append(batch_output['pred'].view(-1).cpu().numpy())
            true_labels.append(batch['labels'].view(-1).cpu().numpy())

    # Precision, Recall, F1, Accuracy

    predicted_labels = np.concatenate(predicted_labels)
    true_labels = np.concatenate(true_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="binary")
    accuracy = accuracy_score(true_labels, predicted_labels)

    loss = loss_tracker.loss
    loss_tracker.on_epoch_end()
    return {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


class DummyProgressBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass


class LossTracker:
    def __init__(
            self,
            ndigits=4,
    ) -> None:
        self.ndigits = ndigits
        self._loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[float] = []

    def update(self, loss_tensor: torch.Tensor):
        loss = loss_tensor.item()
        self._loss = (self._loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1

    def reset(self):
        self._loss = 0
        self.loss_count = 0

    def on_epoch_end(self, reset: bool = True):
        self.history.append(self.loss)
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        return round(float(self._loss), self.ndigits)
