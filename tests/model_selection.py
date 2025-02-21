import os
import re
from collections import defaultdict


def get_best_models() -> dict[int, str]:
    """
    Finds and returns the best model filenames for each fold based on their scores.
    This function searches through the "checkpoints" directory for model files that match
    the pattern "model_fold{fold}_epoch{epoch}_score{score}.pth". It then identifies the
    best model for each fold based on the highest score.
    Returns:
        best_models(dict[int, str]): A dictionary where the keys are fold numbers (int) and the values
                        are the filenames (str) of the best models for those folds.
    """

    model_dir = "checkpoints"
    pattern = re.compile(r"model_fold(\d+)_epoch\d+_score([\d.]+)\.pth")

    best_models: dict[int, str] = {}
    best_scores: defaultdict[int, float] = defaultdict(lambda: float("-inf"))

    # Search for the best model in the model directory
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            fold = int(match.group(1))
            score = float(match.group(2))

            if score > best_scores[fold]:
                best_scores[fold] = score
                best_models[fold] = filename

    for fold, best_model in sorted(best_models.items()):
        print(f"Best model for Fold {fold}: {best_model}")

    return best_models
