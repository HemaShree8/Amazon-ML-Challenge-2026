"""
Validation and evaluation utilities for Business Entity Resolution.

This module evaluates predicted Source-2/Source-3 matches
against the provided training ground truth.
"""

import pandas as pd


def load_ground_truth(path):
    """
    Load the ground-truth TSV file.

    Expected columns:
        source1_entity_id
        matched_entity_ids
    """

    df = pd.read_csv(path, sep="\t")

    # Convert missing values to empty strings.
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")

    return df


def parse_matches(value):
    """
    Convert a comma-separated match string into a set.

    Example:
        'S2-123,S3-456' -> {'S2-123', 'S3-456'}

        '' -> set()
    """

    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


def calculate_f05(precision, recall):
    """
    Calculate F0.5.

    F0.5 gives more importance to precision than recall.
    """

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def evaluate_predictions(predictions, ground_truth):
    """
    Evaluate predictions against ground truth.

    Parameters
    ----------
    predictions : DataFrame
        Must contain:
            source1_entity_id
            matched_entity_ids

    ground_truth : DataFrame
        Must contain:
            source1_entity_id
            matched_entity_ids

    Returns
    -------
    dict
        Precision, recall, F0.5 and TP/FP/FN totals.
    """

    # Convert ground truth to dictionary
    gt_dict = {
        row["source1_entity_id"]: parse_matches(
            row["matched_entity_ids"]
        )
        for _, row in ground_truth.iterrows()
    }

    # Convert predictions to dictionary
    pred_dict = {
        row["source1_entity_id"]: parse_matches(
            row["matched_entity_ids"]
        )
        for _, row in predictions.iterrows()
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for source1_id, expected in gt_dict.items():

        predicted = pred_dict.get(source1_id, set())

        # Correctly predicted matches
        tp = len(expected & predicted)

        # Predicted matches that are not actually matches
        fp = len(predicted - expected)

        # Real matches that the model missed
        fn = len(expected - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Precision
    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = total_tp / (total_tp + total_fp)

    # Recall
    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = total_tp / (total_tp + total_fn)

    f05 = calculate_f05(precision, recall)

    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
    }


def evaluate_file(prediction_path, ground_truth_path):
    """
    Convenience function to evaluate two TSV files.
    """

    predictions = pd.read_csv(
        prediction_path,
        sep="\t"
    )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    return evaluate_predictions(
        predictions,
        ground_truth
    )


if __name__ == "__main__":

    # Example paths.
    # Change these if your repository uses different paths.

    prediction_path = (
        "output/predictions.tsv"
    )

    ground_truth_path = (
        "dataset/train/train_ground_truth.tsv"
    )

    results = evaluate_file(
        prediction_path,
        ground_truth_path
    )

    print("\nEvaluation Results")
    print("------------------")

    for key, value in results.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
