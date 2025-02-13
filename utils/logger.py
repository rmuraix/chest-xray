import abc

import wandb


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def log(self, data: dict, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def finish(self) -> None:
        pass


class WandbLogger(BaseLogger):
    def __init__(
        self, project_name: str, run_name: str | None = None, **kwargs
    ) -> None:
        """
        Initializes the logger with the specified project name and run name.

        See https://docs.wandb.ai/ref/python/init/
        Args:
            project_name (str): The name of the project.
            run_name (str, optional): The name of the run. If not provided, a unique ID will be generated.
            **kwargs: Additional keyword arguments to pass to the wandb.init() function.
        """
        if run_name is None:
            run_name = wandb.util.generate_id()
        wandb.init(project=project_name, name=run_name, **kwargs)

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
