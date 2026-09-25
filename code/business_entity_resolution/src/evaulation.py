"""
Validation and evaluation utilities for Business Entity Resolution.

This module evaluates predicted Source-2/Source-3 matches
against the provided training ground truth.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_ground_truth(path):
    """
    Load the ground-truth TSV file.

    Expected columns:
        source1_entity_id
        matched_entity_ids
    """

    df = pd.read_csv(path, sep="\t")

    # Empty match lists are represented as empty strings
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")

    return df


def create_validation_split(
    ground_truth,
    validation_size=0.2,
    random_state=42
):
    """
    Split ground truth into training and validation sets.

    Parameters
    ----------
    ground_truth : pandas.DataFrame
        Ground-truth data.

    validation_size : float
        Fraction of data used for validation.

    random_state : int
        Ensures the same split is produced every time.

    Returns
    -------
    train_df, validation_df
    """

    train_df, validation_df = train_test_split(
        ground_truth,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True
    )

    return train_df, validation_df


def parse_matches(value):
    """
    Convert a comma-separated match string into a set.

    Example:
        'S2-123,S3-456'
        -> {'S2-123', 'S3-456'}

        ''
        -> set()
    """

    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {
        match_id.strip()
        for match_id in str(value).split(",")
        if match_id.strip()
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
    predictions : pandas.DataFrame
        Must contain:
            source1_entity_id
            matched_entity_ids

    ground_truth : pandas.DataFrame
        Must contain:
            source1_entity_id
            matched_entity_ids

    Returns
    -------
    dict
        TP, FP, FN, precision, recall and F0.5.
    """

    # Convert ground truth into a dictionary
    gt_dict = {
        row["source1_entity_id"]: parse_matches(
            row["matched_entity_ids"]
        )
        for _, row in ground_truth.iterrows()
    }

    # Convert predictions into a dictionary
    pred_dict = {
        row["source1_entity_id"]: parse_matches(
            row["matched_entity_ids"]
        )
        for _, row in predictions.iterrows()
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Compare predictions against ground truth
    for source1_id, expected in gt_dict.items():

        predicted = pred_dict.get(
            source1_id,
            set()
        )

        # Correctly predicted matches
        tp = len(expected & predicted)

        # Predicted but incorrect matches
        fp = len(predicted - expected)

        # Correct matches that were missed
        fn = len(expected - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calculate precision
    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = total_tp / (total_tp + total_fp)

    # Calculate recall
    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = total_tp / (total_tp + total_fn)

    # Calculate F0.5
    f05 = calculate_f05(
        precision,
        recall
    )

    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
    }


def evaluate_file(
    prediction_path,
    ground_truth_path
):
    """
    Evaluate predictions stored in a TSV file.
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

    # Load ground truth
    ground_truth_path = (
        "dataset/train/train_ground_truth.tsv"
    )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    # Create train/validation split
    train_gt, validation_gt = create_validation_split(
        ground_truth
    )

    print("Validation Split")
    print("-----------------")
    print(
        "Total rows:",
        len(ground_truth)
    )
    print(
        "Training rows:",
        len(train_gt)
    )
    print(
        "Validation rows:",
        len(validation_gt)
    )
