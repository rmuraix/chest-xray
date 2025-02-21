from typing import Literal, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, MultilabelAUROC

import wandb
from utils import BaseLogger


class Tester:
    """
    A class to handle the testing of a neural network model.
    Attributes:
        model (nn.Module): The neural network model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_classes (int): Number of classes in the dataset.
        fold (int): The fold number for cross-validation.
        criterion (nn.Module): Loss function.
        device (Union[str, torch.device]): Device to run the model on.
        logger (BaseLogger): Logger for logging metrics.
        log_step (int): Step interval for logging. Default is 100.
        task (Literal["binary", "multiclass", "multilabel"]): Type of classification task. Default is "multilabel".
    Methods:
        test() -> Tuple[wandb.Table, pd.DataFrame, float]:
            Tests the model on the test dataset and return AUC and DataFrame with the results.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        num_classes: int,
        fold: int,
        criterion: nn.Module,
        device: Union[str, torch.device],
        logger: BaseLogger,
        table: wandb.Table,
        log_step: int = 100,
        task: Literal["binary", "multiclass", "multilabel"] = "multilabel",
    ):
        self.model: nn.Module = model
        self.test_loader: DataLoader = test_loader
        self.num_classes: int = num_classes
        self.fold: int = fold
        self.criterion: nn.Module = criterion
        self.device: Union[str, torch.device] = device
        self.logger: BaseLogger = logger
        self.table: wandb.Table = table
        self.log_step: int = log_step
        self.task: Literal["binary", "multiclass", "multilabel"] = task

    def test(self) -> Tuple[wandb.Table, pd.DataFrame, float]:
        self.model.eval()

        metric_auroc: MultilabelAUROC = MultilabelAUROC(
            num_labels=self.num_classes, average="macro"
        ).to(self.device)
        metric_acc: Accuracy = Accuracy(
            task=self.task, num_labels=self.num_classes, average="macro"
        ).to(self.device)
        metric_f1: F1Score = F1Score(
            task=self.task, num_labels=self.num_classes, average="macro"
        ).to(self.device)

        running_loss: float = 0.0
        all_outputs: list = []
        all_paths: list = []

        with (
            torch.no_grad(),
            tqdm.tqdm(self.test_loader, desc="[test]") as t,
        ):
            for i, (inputs, paths, targets) in enumerate(t):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.to(torch.float32)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                outputs = torch.sigmoid(outputs)
                targets = targets.round().long()

                metric_auroc.update(outputs, targets)
                metric_acc.update(outputs, targets)
                metric_f1.update(outputs, targets)

                t.set_postfix(loss=running_loss / (i + 1))

                # Save to list
                all_outputs.extend(outputs.cpu().tolist())
                all_paths.extend(paths)

        avg_loss: float = running_loss / len(self.test_loader)
        avg_auc: torch.Tensor = metric_auroc.compute()
        avg_acc: torch.Tensor = metric_acc.compute()
        avg_f1: torch.Tensor = metric_f1.compute()

        self.table.add_data(
            avg_auc.item(),
            avg_acc.item(),
            avg_f1.item(),
            avg_loss,
        )

        df_results: pd.DataFrame = pd.DataFrame(
            {"Image Index": all_paths, "outputs": all_outputs}
        )

        return self.table, df_results, avg_auc.item()
