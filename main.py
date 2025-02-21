import hydra
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from omegaconf import DictConfig, OmegaConf
from rich import print
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader, Subset

import wandb
from datasets import NihDataset, test_transform, train_transform
from models import DenseNetMultiLabel
from tests import Tester, get_best_models, save_weighted_final_result
from train import Trainer
from utils import WandbLogger, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    batch_size = cfg.train.batch_size

    if cfg.mode == "train":
        dataset = NihDataset(mode="train")
        labels = dataset.get_labels()

        n_splits = cfg.train.n_splits
        mskf = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )

        for fold, (train_idx, val_idx) in enumerate(mskf.split(dataset, labels)):
            print(f"[bold magenta]Fold {fold + 1}/{n_splits}[/bold magenta]")
            criterion = torch.nn.BCEWithLogitsLoss()
            model = DenseNetMultiLabel(num_classes=14).to(cfg.device)
            optimizer = RAdamScheduleFree(
                model.parameters(),
                lr=cfg.train.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-3,
            )
            logger = WandbLogger(cfg)
            logger.watch(model, log="all", log_freq=100)

            # Create Subsets for the current fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_subset.dataset.set_transform(train_transform())  # type: ignore
            val_subset.dataset.set_transform(test_transform())  # type: ignore

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
            )

            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=cfg.device,
                logger=logger,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=14,
                max_epochs=cfg.train.epochs,
                fold=fold,
                top_k=cfg.train.top_k,
                patience=cfg.train.patience,
            )

            trainer.fit()

            logger.finish()

    elif cfg.mode == "test":
        fold_results = []
        auc_scores = []

        best_models = get_best_models()

        logger = WandbLogger(cfg)

        table: wandb.Table = wandb.Table(
            columns=["auc", "accuracy", "f1", "loss"],
        )

        for fold in range(cfg.train.n_splits):
            print(
                f"[bold magenta]Test Fold {fold + 1}/{cfg.train.n_splits}[/bold magenta]"
            )
            model = DenseNetMultiLabel(num_classes=14, weights=None).to(cfg.device)
            model.load_state_dict(
                torch.load(f"checkpoints/{best_models[fold]}")["model"]
            )

            dataset = NihDataset(mode="test", transform=test_transform())

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
            )

            tester = Tester(
                model=model,
                device=cfg.device,
                test_loader=loader,
                num_classes=14,
                fold=fold,
                logger=logger,
                criterion=torch.nn.BCEWithLogitsLoss(),
                table=table,
            )

            table, df_results, auc = tester.test()
            df_results.to_csv(f"predictions_fold{fold + 1}.csv", index=False)
            fold_results.append(df_results)
            auc_scores.append(auc)

        logger.log({"test_metrics": table})

        fold_results = [
            pd.read_csv(f"predictions_fold{fold + 1}.csv") for fold in range(5)
        ]
        save_weighted_final_result(
            fold_results, auc_scores, output_path="predictions.csv"
        )

        logger.finish()

    else:
        raise ValueError("Invalid mode.")
    return


if __name__ == "__main__":
    set_seed(42)
    main()
