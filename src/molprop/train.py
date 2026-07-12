"""Shared training loop, early stopping, and metrics for any encoder."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from molprop.data import PropertyDataset, collate_batch


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, and R^2."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


@dataclass
class Trainer:
    """Trains any :class:`~molprop.models.MolEncoder` with standardized targets.

    Targets are standardized using the training-set mean and standard deviation,
    optimized with Adam under an MSE loss, and de-standardized at prediction time.
    Passing ``patience`` enables early stopping on the validation RMSE and
    restores the best-scoring weights when training stops.

    Attributes:
        model: The encoder to train.
        lr: Adam learning rate.
        weight_decay: Adam L2 penalty.
        device: Torch device string.
        patience: Epochs of no validation improvement tolerated before stopping.
            Zero disables early stopping. Ignored when no validation set is given.
        history: Per-epoch ``{"train": [...], "val": [...]}`` losses; ``train`` is
            the standardized MSE and ``val`` is the de-standardized RMSE.
        best_epoch: One-based epoch of the lowest validation RMSE, or ``-1``.
        stopped_epoch: One-based epoch at which training actually stopped.
    """

    model: torch.nn.Module
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    patience: int = 0
    target_mean: float = 0.0
    target_std: float = 1.0
    history: dict[str, list[float]] = field(default_factory=lambda: {"train": [], "val": []})
    best_epoch: int = -1
    stopped_epoch: int = -1

    def _loader(self, dataset: PropertyDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
        )

    def fit(
        self,
        train_set: PropertyDataset,
        val_set: PropertyDataset | None = None,
        epochs: int = 40,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Trainer:
        """Train the encoder, recording per-epoch history.

        Args:
            train_set: Training split; its target statistics set the standardizer.
            val_set: Optional validation split used for early stopping and history.
            epochs: Maximum number of epochs.
            batch_size: Minibatch size.
            verbose: If True, print per-epoch train MSE and validation RMSE.

        Returns:
            The trainer itself, for chaining.
        """
        self.model.to(self.device)
        self.target_mean = float(train_set.targets.mean())
        self.target_std = float(train_set.targets.std()) or 1.0
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        loader = self._loader(train_set, batch_size, shuffle=True)
        has_val = val_set is not None and len(val_set) > 0

        best_rmse = float("inf")
        best_state: dict | None = None
        epochs_since_best = 0

        for epoch in range(epochs):
            self.model.train()
            total, n = 0.0, 0
            for batch in loader:
                batch = batch.to(self.device)
                target = (batch.y - self.target_mean) / self.target_std
                opt.zero_grad()
                loss = loss_fn(self.model(batch), target)
                loss.backward()
                opt.step()
                total += loss.item() * batch.y.shape[0]
                n += batch.y.shape[0]
            train_loss = total / max(n, 1)
            self.history["train"].append(train_loss)
            val_rmse = float("nan")
            if has_val:
                val_rmse = self.evaluate(val_set, batch_size)["rmse"]
            self.history["val"].append(val_rmse)
            if verbose:
                print(f"epoch {epoch + 1:3d}  train_mse={train_loss:.4f}  val_rmse={val_rmse:.4f}")

            self.stopped_epoch = epoch + 1
            if has_val and val_rmse < best_rmse:
                best_rmse = val_rmse
                self.best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if has_val and self.patience > 0 and epochs_since_best >= self.patience:
                    if verbose:
                        print(f"early stop at epoch {epoch + 1} (best {self.best_epoch})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, dataset: PropertyDataset, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        preds: list[np.ndarray] = []
        for batch in self._loader(dataset, batch_size, shuffle=False):
            batch = batch.to(self.device)
            out = self.model(batch) * self.target_std + self.target_mean
            preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0) if preds else np.zeros((0,))

    def evaluate(self, dataset: PropertyDataset, batch_size: int = 64) -> dict[str, float]:
        return regression_metrics(dataset.targets, self.predict(dataset, batch_size))
