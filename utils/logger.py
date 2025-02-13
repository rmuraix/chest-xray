import abc

from omegaconf import DictConfig, OmegaConf, errors

import wandb


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def log(self, data: dict, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def finish(self) -> None:
        pass


class WandbLogger(BaseLogger):
    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        """
        Initializes the WandbLogger with the specified configuration.

        This method sets up the Weights and Biases (wandb) logger with the provided project name and run name from the configuration.
        If the run name is not provided, a unique ID will be generated.

        See https://docs.wandb.ai/ref/python/init/ and https://docs.wandb.ai/guides/integrations/hydra/

        Args:
            cfg (DictConfig): The hydra configuration object containing wandb settings.
            **kwargs: Additional keyword arguments to pass to the wandb.init() function.

        Returns:
            None
        """
        # if cfg.wandb.run_name is not provided, generate a unique run name
        try:
            run_name = cfg.wandb.run_name
        except errors.ConfigAttributeError:
            run_name = wandb.util.generate_id()

        wandb.init(project=cfg.wandb.project, name=run_name, **kwargs)

        # Track the configuration
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.config.update(config_dict)

    def log(self, data: dict, **kwargs) -> None:
        """
        Logs the provided data to Weights and Biases (wandb).

        See https://docs.wandb.ai/ref/python/log

        Args:
            data (dict): A dictionary containing the data to be logged.
            **kwargs: Additional keyword arguments to be passed to the wandb.log function.

        Returns:
            None
        """
        wandb.log(data, **kwargs)

    def finish(self) -> None:
        """
        Finish the current Weights and Biases (wandb) run.

        This method finalizes the current wandb run, ensuring that all data is properly logged and the run is closed.

        See https://docs.wandb.ai/ref/python/finish/
        """
        wandb.finish()
