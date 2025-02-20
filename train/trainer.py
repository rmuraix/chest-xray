import heapq
import os
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, MultilabelAUROC

from utils import BaseLogger


class Trainer:
    """
    Trainer class for training and validating a PyTorch model.

    Attributes:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (Optional[DataLoader]): DataLoader for the validation dataset.
        num_classes (int): Number of classes in the dataset.
        criterion (nn.Module): Loss function.
        max_epochs (int): Maximum number of epochs to train the model.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (Union[str, torch.device]): Device to run the model on (e.g., 'cpu' or 'cuda').
        logger (BaseLogger): Logger for logging training and validation information.
        top_k (int): Number of top models to keep.
        higher_is_better (bool): Whether a higher score is better (e.g., accuracy) or lower (e.g., loss).
        save_dir (str): Directory to save the model checkpoints.
        patience (int): Number of epochs to wait for early stopping.
        score_function (Optional[Callable[[torch.Tensor, torch.Tensor], float]]): Custom score function.
            If None, uses accuracy by default.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        fold: int,
        criterion: nn.Module,
        max_epochs: int,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device],
        logger: BaseLogger,
        log_step: int = 100,
        task: Literal["binary", "multiclass", "multilabel"] = "multilabel",
        top_k: int = 5,
        save_dir: str = "checkpoints",
        higher_is_better: bool = True,
        patience: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.fold = fold
        self.max_epochs = max_epochs
        self.task = task
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.log_step = log_step
        self.top_k = top_k
        self.save_dir = save_dir
        self.patience = patience

        self.scaler = torch.amp.GradScaler()

        # For saving the models
        self.comparator = 1 if higher_is_better else -1
        self.saved_models: list[tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)

        # Early Stopping Parameters
        self.best_score: float | None = None
        self.epochs_no_improve: int = 0

        # For logging
        self.step: int = 0

    def save(self, score: float, epoch: int) -> None:
        """
        Save the model checkpoint with its score and keep only top-k models.

        Args:
            score (float): The evaluation score (e.g., validation accuracy).
            epoch (int): The epoch number.
        """
        checkpoint_path = os.path.join(
            self.save_dir,
            f"model_fold{self.fold}_epoch{epoch + 1}_score{score:.4f}.pth",
        )
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

        # Save the model checkpoint and score
        heapq.heappush(self.saved_models, (score * self.comparator, checkpoint_path))

        if len(self.saved_models) <= self.top_k:
            print(f"Saved model: {checkpoint_path} with score: {score:.4f}")
            return

        worst_score, worst_path = heapq.heappop(self.saved_models)
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(
                f"Removed worst model: {worst_path} "
                f"with score: {worst_score / self.comparator:.4f}"
            )
        print(f"Saved model: {checkpoint_path} with score: {score:.4f}")

    def check_early_stopping(self, score: float) -> bool:
        """
        Check if early stopping criteria are met based on the provided score.

        If the current score is better than the best score recorded so far,
        update and reset patience counter. Otherwise, increment the counter.
        If the counter reaches the patience threshold, returns True.

        Args:
            score (float): The current score to evaluate.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if self.best_score is None or (score * self.comparator) > (
            self.best_score * self.comparator
        ):
            self.best_score = score
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        return self.epochs_no_improve >= self.patience

    def train_on_epoch(self, epoch: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        self.optimizer.train()
        metric_auroc = MultilabelAUROC(num_labels=self.num_classes, average="macro").to(
            self.device
        )
        metric_acc = Accuracy(
            task="multilabel", num_labels=self.num_classes, average="macro"
        ).to(self.device)
        metric_f1 = F1Score(
            task="multilabel", num_labels=self.num_classes, average="macro"
        ).to(self.device)
        running_loss = 0.0

        with tqdm.tqdm(self.train_loader, desc=f"[train] Epoch {epoch + 1}") as t:
            for i, (inputs, _, targets) in enumerate(t):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.to(torch.float32)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)

                self.scaler.update()

                running_loss += loss.item()

                # Compute the score using the provided function or default to accuracy
                outputs = torch.sigmoid(outputs)
                metric_auroc.update(outputs, targets.round().long())
                metric_acc.update(outputs, targets.round().long())
                metric_f1.update(outputs, targets.round().long())

                self.step += 1

                if i % self.log_step == 0:
                    self.logger.log(
                        {
                            "train_loss": running_loss / (i + 1),
                        },
                        step=self.step,
                    )

                t.set_postfix(loss=running_loss / (i + 1))

        avg_loss = running_loss / len(self.train_loader)
        avg_auroc = metric_auroc.compute()
        avg_acc = metric_acc.compute()
        avg_f1 = metric_f1.compute()

        self.logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_auc": avg_auroc,
                "train_acc": avg_acc,
                "train_f1": avg_f1,
            },
            step=self.step,
        )
        return avg_loss

    def validate_on_epoch(self, epoch: int) -> tuple[float, float | None]:
        """
        Validates the model on the validation dataset for a given epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            (float, float | None): The average validation loss and the validation score.
                If val_loader is None, returns (0.0, None).
        """
        if self.val_loader is None:
            return 0.0, None

        self.model.eval()
        self.optimizer.eval()
        metric_auroc = MultilabelAUROC(num_labels=self.num_classes, average="macro").to(
            self.device
        )
        metric_acc = Accuracy(
            task="multilabel", num_labels=self.num_classes, average="macro"
        ).to(self.device)
        metric_f1 = F1Score(
            task="multilabel", num_labels=self.num_classes, average="macro"
        ).to(self.device)

        running_loss = 0.0

        with (
            torch.no_grad(),
            tqdm.tqdm(self.val_loader, desc=f"[valid] Epoch {epoch + 1}") as t,
        ):
            for i, (inputs, _, targets) in enumerate(t):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.to(torch.float32)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                # Compute the score using the provided function or default to accuracy
                outputs = torch.sigmoid(outputs)
                metric_auroc.update(outputs, targets.round().long())
                metric_acc.update(outputs, targets.round().long())
                metric_f1.update(outputs, targets.round().long())

                self.step += 1

                if i % self.log_step == 0:
                    self.logger.log(
                        {
                            "train_loss": running_loss / (i + 1),
                        },
                        step=self.step,
                    )

                t.set_postfix(loss=running_loss / (i + 1))

        avg_loss = running_loss / len(self.val_loader)
        avg_auc = metric_auroc.compute()
        avg_acc = metric_acc.compute()
        avg_f1 = metric_f1.compute()

        self.logger.log(
            {
                "epoch": epoch + 1,
                "val_loss": avg_loss,
                "val_auc": avg_auc,
                "val_acc": avg_acc,
                "val_f1": avg_f1,
            },
            step=self.step,
        )

        return avg_loss, avg_auc

    def fit(self) -> None:
        """
        Trains the model for multiple epochs or until early stopping criteria are met.
        """
        for epoch in range(self.max_epochs):
            self.train_on_epoch(epoch)
            val_loss, val_score = self.validate_on_epoch(epoch)

            if val_score is not None:
                self.save(val_score, epoch)

                if self.check_early_stopping(val_score):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            if epoch + 1 == self.max_epochs:
                print(f"Reached maximum number of epochs: {self.max_epochs}")
                break

    def load_best_model(self) -> None:
        """
        Loads the best-scored model's state_dict back into self.model.
        If 'saved_models' is empty, does nothing.
        """
        if not self.saved_models:
            print("No saved model found.")
            return

        # self.saved_models is a min-heap, so the best model is the last element
        best = max(self.saved_models, key=lambda x: x[0] * self.comparator)
        best_model_path = best[1]
        checkpoint = torch.load(best_model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded best model from {best_model_path}")
