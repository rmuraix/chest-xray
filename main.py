import hydra
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from omegaconf import DictConfig, OmegaConf
from rich import print
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader, Subset

from datasets import NihDataset, test_transform, train_transform
from models import DenseNetMultiLabel
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
        raise NotImplementedError("Test mode not implemented yet.")
    else:
        raise ValueError("Invalid mode.")
    return


if __name__ == "__main__":
    set_seed(42)
    main()
