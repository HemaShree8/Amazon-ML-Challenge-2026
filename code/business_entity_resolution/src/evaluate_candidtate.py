"""
Evaluate candidate generation against training ground truth.

This checks whether the blocking/candidate-generation stage
contains the true S2/S3 matches.

Candidate recall:

    true matches found in candidates
    --------------------------------
          total true matches
"""

import pandas as pd


def parse_ids(value):
    """
    Convert comma-separated IDs into a set.

    Example:
        'S2-123,S3-456'
        -> {'S2-123', 'S3-456'}

        empty value
        -> set()
    """

    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


def load_ground_truth(path):
    """
    Load training ground truth.
    """

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )


def load_candidates(path):
    """
    Load generated candidate pairs.
    """

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )


def evaluate_candidate_recall(
    ground_truth,
    candidates
):
    """
    Compare true matches against generated candidates.
    """

    # Make dictionaries:
    #
    # S1 ID -> set of true matches
    #
    gt_dict = {
        row["source1_entity_id"]:
            parse_ids(row["matched_entity_ids"])
        for _, row in ground_truth.iterrows()
    }

    # S1 ID -> set of generated candidates
    candidate_dict = {
        row["source1_entity_id"]:
            parse_ids(row["candidate_entity_ids"])
        for _, row in candidates.iterrows()
    }

    total_true_matches = 0
    found_matches = 0
    missed_matches = 0

    missed_examples = []

    for source1_id, true_matches in gt_dict.items():

        generated_candidates = candidate_dict.get(
            source1_id,
            set()
        )

        # True matches that were successfully generated
        found = true_matches & generated_candidates

        # True matches that blocking missed
        missed = true_matches - generated_candidates

        total_true_matches += len(true_matches)
        found_matches += len(found)
        missed_matches += len(missed)

        # Store a few examples for debugging
        if missed and len(missed_examples) < 20:

            missed_examples.append({
                "source1_entity_id": source1_id,
                "missed_matches": ",".join(
                    sorted(missed)
                ),
                "generated_candidate_count":
                    len(generated_candidates)
            })

    if total_true_matches > 0:

        recall = (
            found_matches
            / total_true_matches
        )

    else:

        recall = 0.0

    return {
        "total_true_matches":
            total_true_matches,

        "found_matches":
            found_matches,

        "missed_matches":
            missed_matches,

        "candidate_recall":
            recall,

        "missed_examples":
            missed_examples
    }


def main():

    ground_truth_path = (
        "dataset/train/train_ground_truth.tsv"
    )

    candidate_path = (
        "output/candidate_pairs.tsv"
    )

    print("Loading ground truth...")

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    print("Loading candidate pairs...")

    candidates = load_candidates(
        candidate_path
    )

    print(
        "\nGround truth rows:",
        len(ground_truth)
    )

    print(
        "Candidate rows:",
        len(candidates)
    )

    results = evaluate_candidate_recall(
        ground_truth,
        candidates
    )

    print("\n" + "=" * 60)
    print("CANDIDATE RECALL")
    print("=" * 60)

    print(
        "Total true matches:",
        f"{results['total_true_matches']:,}"
    )

    print(
        "True matches found:",
        f"{results['found_matches']:,}"
    )

    print(
        "True matches missed:",
        f"{results['missed_matches']:,}"
    )

    print(
        "Candidate recall:",
        f"{results['candidate_recall']:.4%}"
    )

    print("\nMissed examples:")

    for example in results["missed_examples"]:

        print(
            f"\nS1: "
            f"{example['source1_entity_id']}"
        )

        print(
            f"Missed: "
            f"{example['missed_matches']}"
        )

        print(
            f"Generated candidates: "
            f"{example['generated_candidate_count']}"
        )


if __name__ == "__main__":
    main()
