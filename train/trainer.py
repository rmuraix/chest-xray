import heapq
import os
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torchmetrics
import tqdm
from torch.utils.data import DataLoader

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
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        criterion: nn.Module,
        max_epochs: int,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device],
        logger: BaseLogger,
        task: str = "multiclass",
        top_k: int = 5,
        save_dir: str = "checkpoints",
        higher_is_better: bool = False,
        patience: int = 10,
        score_function: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.task = task
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.top_k = top_k
        self.save_dir = save_dir
        self.patience = patience
        self.score_function = score_function

        # For saving the models
        self.comparator = 1 if higher_is_better else -1
        self.saved_models: list[tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)

        # Accuracy metric instance created once for performance improvement
        self.accuracy_metric = torchmetrics.classification.Accuracy(
            task=task, num_classes=num_classes
        )

        # Early Stopping Parameters
        self.patience = patience
        self.best_score: float | None = None
        self.epochs_no_improve: int = 0

    def save(self, score: float, epoch: int):
        """
        Save the model checkpoint with its score and keep only top-k models.

        Args:
            score (float): The evaluation score (e.g., validation loss or accuracy).
            epoch (int): The epoch number.
        """
        checkpoint_path = os.path.join(
            self.save_dir, f"model_epoch_{epoch}_score_{score:.4f}.pth"
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
                f"Removed worst model: {worst_path} with score: {worst_score / self.comparator:.4f}"
            )

        print(f"Saved model: {checkpoint_path} with score: {score:.4f}")

    def compute_accuracy(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the accuracy of the model.

        Args:
            outputs (torch.Tensor): The model predictions.
            targets (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The accuracy of the model.
        """
        return self.accuracy_metric(outputs, targets)

    def check_early_stopping(self, score: float) -> bool:
        """
        Check if early stopping criteria are met based on the provided score.

        This method compares the current score with the best score recorded so far.
        If the current score is better, it updates the best score and resets the
        counter for epochs without improvement. If the current score is not better,
        it increments the counter for epochs without improvement. If the counter
        reaches the patience threshold, early stopping criteria are met.

        Args:
            score (float): The current score to evaluate.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if (
            self.best_score is None
            or score * self.comparator > self.best_score * self.comparator
        ):
            self.best_score = score
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        return self.epochs_no_improve >= self.patience

    def train_on_epoch(self, epoch: int) -> torch.Tensor:
        """
        Trains the model for one epoch.
        Args:
            epoch (int): The current epoch number.
        Returns:
            torch.Tensor: The average loss for the epoch.
        """
        self.model.train()
        self.accuracy_metric.reset()
        running_loss = 0.0
        total_score = 0.0
        num_batches = 0

        with tqdm.tqdm(self.train_loader, desc=f"[train] Epoch {epoch + 1}") as t:
            for i, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Compute the score using the provided function or default to accuracy
                if self.score_function:
                    score = self.score_function(outputs, targets)
                else:
                    score = self.accuracy_metric(outputs, targets).item()

                total_score += score
                num_batches += 1

                t.set_postfix(
                    loss=running_loss / (i + 1), score=total_score / num_batches
                )

        avg_loss = running_loss / len(self.train_loader)
        avg_score = total_score / num_batches

        self.logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_score": avg_score,
            }
        )
        return avg_loss

    def validate_on_epoch(self, epoch: int) -> torch.Tensor:
        """
        Validates the model on the validation dataset for a given epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            torch.Tensor: The average loss over the validation dataset.

        Notes:
            - If `self.val_loader` is None, returns 0.0.
            - Sets the model to evaluation mode.
            - Uses `torch.no_grad()` to disable gradient computation.
            - Displays a progress bar with the current loss using `tqdm`.
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        self.accuracy_metric.reset()
        running_loss = 0.0

        with (
            torch.no_grad(),
            tqdm.tqdm(self.val_loader, desc=f"[valid] Epoch {epoch + 1}") as t,
        ):
            for i, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                self.accuracy_metric.update(outputs, targets)
                t.set_postfix(loss=running_loss / (i + 1))

        avg_loss = running_loss / len(self.val_loader)
        val_acc = self.accuracy_metric.compute().item()

        # Compute the score using the provided function or default to accuracy
        if self.score_function:
            val_score = self.score_function(outputs, targets)
        else:
            val_score = val_acc  # Default to accuracy

        self.logger.log(
            {
                "epoch": epoch + 1,
                "val_loss": avg_loss,
                "val_accuracy": val_acc,
                "val_score": val_score,
            }
        )

        self.save(val_score, epoch)

        if self.check_early_stopping(val_score):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            return None

        return avg_loss

    def fit(self):
        """
        Trains the model for multiple epochs until early stopping criteria are met.
        """
        for epoch in range(self.patience):
            self.train_on_epoch(epoch)
            val_loss = self.validate_on_epoch(epoch)

            self.save(val_loss, epoch)

            if val_loss is None:
                # Early stopping triggered
                break
            if epoch + 1 == self.max_epochs:
                print(f"Reached maximum number of epochs: {self.max_epochs}")
                break
