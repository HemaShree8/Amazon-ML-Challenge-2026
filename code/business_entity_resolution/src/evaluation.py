"""
Validation and evaluation utilities for Business Entity Resolution.

The challenge evaluates matching quality per Source-1 entity,
including singleton entities, using F0.5.

This module provides:
    - ground-truth loading
    - validation split
    - prediction parsing
    - per-entity precision/recall/F0.5
    - macro F0.5
    - overall TP/FP/FN statistics
    - evaluation of prediction TSV files
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(path):
    """
    Load ground-truth TSV.

    Expected columns:
        source1_entity_id
        matched_entity_ids
    """

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing ground-truth columns: {sorted(missing)}"
        )

    df["matched_entity_ids"] = (
        df["matched_entity_ids"]
        .fillna("")
        .astype(str)
    )

    return df


# ============================================================
# VALIDATION SPLIT
# ============================================================

def create_validation_split(
    ground_truth,
    validation_size=0.20,
    random_state=42,
):
    """
    Split Source-1 entities into training and validation sets.

    The split is performed at Source-1 entity level, so an
    entity and all of its matches remain together.
    """

    train_df, validation_df = train_test_split(
        ground_truth,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
    )


# ============================================================
# MATCH PARSING
# ============================================================

def parse_matches(value):
    """
    Convert comma-separated match IDs into a set.

    Example:
        "S2-123,S3-456"
        -> {"S2-123", "S3-456"}

        ""
        -> set()
    """

    if value is None:
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        match_id.strip()
        for match_id in value.split(",")
        if match_id.strip()
    }


# ============================================================
# F0.5
# ============================================================

def calculate_f05(precision, recall):
    """
    Calculate F0.5.

    F0.5 weights precision more heavily than recall.
    """

    if precision == 0.0 and recall == 0.0:
        return 0.0

    denominator = (
        0.25 * precision
        + recall
    )

    if denominator == 0.0:
        return 0.0

    return (
        1.25
        * precision
        * recall
        / denominator
    )


# ============================================================
# PER-ENTITY EVALUATION
# ============================================================

def evaluate_entity(expected, predicted):
    """
    Evaluate one Source-1 entity.

    Returns:
        TP
        FP
        FN
        precision
        recall
        f0.5
    """

    true_positive = len(
        expected & predicted
    )

    false_positive = len(
        predicted - expected
    )

    false_negative = len(
        expected - predicted
    )

    # --------------------------------------------------------
    # Singleton / no-match entity
    # --------------------------------------------------------

    if not expected:

        if not predicted:
            # Correctly predicted that this entity has no
            # Source-2/Source-3 matches.
            return {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f0.5": 1.0,
            }

        # Singleton incorrectly received a match.
        return {
            "true_positives": 0,
            "false_positives": false_positive,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    # --------------------------------------------------------
    # Entity has expected matches
    # --------------------------------------------------------

    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = (
            true_positive
            / (true_positive + false_positive)
        )

    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = (
            true_positive
            / (true_positive + false_negative)
        )

    f05 = calculate_f05(
        precision,
        recall,
    )

    return {
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
    }


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate_predictions(
    predictions,
    ground_truth,
    return_details=False,
):
    """
    Evaluate predictions against ground truth.

    The primary metric is macro-average F0.5 across
    Source-1 entities.

    Singleton entities are included:
        correct empty prediction -> F0.5 = 1
        incorrect match          -> F0.5 = 0
    """

    gt_dict = {
        row["source1_entity_id"]:
            parse_matches(row["matched_entity_ids"])
        for _, row in ground_truth.iterrows()
    }

    pred_dict = {
        row["source1_entity_id"]:
            parse_matches(row["matched_entity_ids"])
        for _, row in predictions.iterrows()
    }

    # --------------------------------------------------------
    # Check duplicate prediction IDs
    # --------------------------------------------------------

    if predictions[
        "source1_entity_id"
    ].duplicated().any():

        raise ValueError(
            "Predictions contain duplicate "
            "source1_entity_id values."
        )

    # --------------------------------------------------------
    # Evaluate every ground-truth Source-1 entity
    # --------------------------------------------------------

    total_tp = 0
    total_fp = 0
    total_fn = 0

    entity_scores = []

    for source1_id, expected in gt_dict.items():

        predicted = pred_dict.get(
            source1_id,
            set(),
        )

        metrics = evaluate_entity(
            expected,
            predicted,
        )

        total_tp += metrics[
            "true_positives"
        ]

        total_fp += metrics[
            "false_positives"
        ]

        total_fn += metrics[
            "false_negatives"
        ]

        entity_scores.append(
            {
                "source1_entity_id": source1_id,
                "expected_count": len(expected),
                "predicted_count": len(predicted),
                **metrics,
            }
        )

    # --------------------------------------------------------
    # Overall micro statistics
    # --------------------------------------------------------

    if total_tp + total_fp == 0:
        overall_precision = 0.0
    else:
        overall_precision = (
            total_tp
            / (total_tp + total_fp)
        )

    if total_tp + total_fn == 0:
        overall_recall = 0.0
    else:
        overall_recall = (
            total_tp
            / (total_tp + total_fn)
        )

    overall_f05 = calculate_f05(
        overall_precision,
        overall_recall,
    )

    # --------------------------------------------------------
    # Challenge-style macro F0.5
    # --------------------------------------------------------

    if entity_scores:
        macro_f05 = sum(
            row["f0.5"]
            for row in entity_scores
        ) / len(entity_scores)
    else:
        macro_f05 = 0.0

    result = {
        "source1_entities": len(entity_scores),

        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,

        "precision": overall_precision,
        "recall": overall_recall,
        "micro_f0.5": overall_f05,

        "macro_f0.5": macro_f05,
    }

    if return_details:
        details = pd.DataFrame(
            entity_scores
        )

        return result, details

    return result


# ============================================================
# EVALUATE FILE
# ============================================================

def evaluate_file(
    prediction_path,
    ground_truth_path,
    details_output=None,
):
    """
    Evaluate a prediction TSV file.
    """

    predictions = pd.read_csv(
        prediction_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    result, details = evaluate_predictions(
        predictions,
        ground_truth,
        return_details=True,
    )

    if details_output:
        details_path = Path(
            details_output
        )

        details_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        details.to_csv(
            details_path,
            sep="\t",
            index=False,
        )

    return result


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):
    """
    Print evaluation metrics.
    """

    print()
    print("=" * 60)
    print("MATCHING EVALUATION")
    print("=" * 60)

    print(
        f"Source-1 entities : "
        f"{results['source1_entities']:,}"
    )

    print(
        f"True positives    : "
        f"{results['true_positives']:,}"
    )

    print(
        f"False positives   : "
        f"{results['false_positives']:,}"
    )

    print(
        f"False negatives   : "
        f"{results['false_negatives']:,}"
    )

    print(
        f"Precision         : "
        f"{results['precision']:.6f}"
    )

    print(
        f"Recall            : "
        f"{results['recall']:.6f}"
    )

    print(
        f"Micro F0.5        : "
        f"{results['micro_f0.5']:.6f}"
    )

    print(
        f"Macro F0.5        : "
        f"{results['macro_f0.5']:.6f}"
    )

    print("=" * 60)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate entity-resolution predictions."
    )

    parser.add_argument(
        "--predictions",
        default=None,
        help="Prediction TSV file.",
    )

    parser.add_argument(
        "--ground-truth",
        default=(
            "dataset/train/"
            "train_ground_truth.tsv"
        ),
        help="Ground-truth TSV file.",
    )

    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional per-entity evaluation TSV.",
    )

    parser.add_argument(
        "--validation-split",
        action="store_true",
        help="Show the 80/20 validation split.",
    )

    args = parser.parse_args()

    ground_truth = load_ground_truth(
        args.ground_truth
    )

    if args.validation_split:

        train_gt, validation_gt = (
            create_validation_split(
                ground_truth
            )
        )

        print(
            f"Total rows      : "
            f"{len(ground_truth):,}"
        )

        print(
            f"Training rows   : "
            f"{len(train_gt):,}"
        )

        print(
            f"Validation rows : "
            f"{len(validation_gt):,}"
        )

    if args.predictions:

        results = evaluate_file(
            args.predictions,
            args.ground_truth,
            args.details_output,
        )

        print_results(results)
