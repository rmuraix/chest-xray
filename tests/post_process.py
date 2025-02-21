import os

import numpy as np
import pandas as pd
from rich import print
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def save_weighted_final_result(
    fold_results,
    auc_scores,
    test_csv_path="datasets/data/nih/test.csv",
    output_path="predictions.csv",
):
    if len(fold_results) != len(auc_scores):
        raise ValueError("The lengths of fold_results and auc_scores do not match.")

    # Calculate normalized AUC scores (for weighting)
    auc_scores = np.array(auc_scores)
    weights = auc_scores / auc_scores.sum()

    combined_df = fold_results[0].copy()
    combined_df["Image Index"] = combined_df["Image Index"].apply(
        lambda x: os.path.basename(x)
    )

    # Weighted average of predicted values by AUC scores
    num_samples = len(combined_df)
    num_labels = len(eval(combined_df.iloc[0]["outputs"]))
    # Initialize (N, 14) matrix
    weighted_outputs = np.zeros((num_samples, num_labels), dtype=float)

    for df, weight in zip(fold_results, weights):
        df["outputs"] = df["outputs"].apply(eval)
        df["Image Index"] = df["Image Index"].apply(lambda x: os.path.basename(x))
        weighted_outputs += np.vstack(df["outputs"]) * weight

    combined_df.drop(columns=["outputs"], inplace=True)

    df_gt = pd.read_csv(test_csv_path)
    df_gt["Image Index"] = df_gt["Image Index"].apply(lambda x: os.path.basename(x))

    # Change column names for predictive labels
    label_names = df_gt.columns[1:].tolist()
    pred_label_names = [f"pred_{label}" for label in label_names]
    weighted_outputs_df = pd.DataFrame(weighted_outputs, columns=pred_label_names)

    final_df = df_gt.merge(combined_df, on="Image Index", how="left")
    final_df = final_df.merge(weighted_outputs_df, left_index=True, right_index=True)

    final_df.to_csv(output_path, index=False)
    print(f"Weighted predictions saved to {output_path}")

    label_columns = [
        col
        for col in final_df.columns
        if not col.startswith("pred_") and col != "Image Index"
    ]
    pred_columns = [f"pred_{col}" for col in label_columns]

    auc_scores = []
    for label, pred in zip(label_columns, pred_columns):
        if len(final_df[label].unique()) > 1:
            auc_scores.append(roc_auc_score(final_df[label], final_df[pred]))
    macro_auc = sum(auc_scores) / len(auc_scores)

    binary_preds = final_df[pred_columns] >= 0.5
    macro_accuracy = sum(
        accuracy_score(final_df[label], binary_preds[pred])
        for label, pred in zip(label_columns, pred_columns)
    ) / len(label_columns)
    macro_f1 = sum(
        f1_score(final_df[label], binary_preds[pred])
        for label, pred in zip(label_columns, pred_columns)
    ) / len(label_columns)

    print(f"[bold green]Macro AUC: {macro_auc:.4f}[/bold green]")
    print(f"[bold green]Macro Accuracy: {macro_accuracy:.4f}[/bold green]")
    print(f"[bold green]Macro F1: {macro_f1:.4f}[/bold green]")
