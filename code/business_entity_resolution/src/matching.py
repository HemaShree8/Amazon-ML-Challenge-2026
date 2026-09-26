"""
Business Entity Resolution - Matching Stage

Input:
    V2-A candidate_pairs_v2a.tsv

Output:
    matching_results.tsv

The matcher compares Source-1 records against only the
candidate Source-2/Source-3 records produced by V2-A.

Shared preprocessing utilities are imported from preprocessing.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from preprocessing import (
    normalize_name,
    normalize_address,
    name_tokens,
    address_tokens,
    address_signature,
    address_signature_without_numbers,
    jaccard_similarity,
    containment_similarity,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_THRESHOLD = 0.80

# Candidates with extremely weak evidence are rejected.
MIN_NAME_EVIDENCE = 0.20
MIN_ADDRESS_EVIDENCE = 0.20


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_string(value):
    """Convert a value to a clean string."""

    if value is None:
        return ""

    return str(value).strip()


def same_country(country_a, country_b):
    """Return True when country labels agree."""

    return (
        safe_string(country_a).casefold()
        == safe_string(country_b).casefold()
    )


def parse_match_ids(value):
    """Convert comma-separated IDs into a set."""

    value = safe_string(value)

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


# ============================================================
# NAME FEATURES
# ============================================================

def calculate_name_features(name_a, name_b):
    """
    Calculate business-name similarity features.
    """

    normalized_a = normalize_name(name_a)
    normalized_b = normalize_name(name_b)

    tokens_a = name_tokens(name_a)
    tokens_b = name_tokens(name_b)

    if not normalized_a or not normalized_b:
        return {
            "ratio": 0.0,
            "jaccard": 0.0,
            "containment": 0.0,
            "exact": 0.0,
        }

    ratio = (
        fuzz.ratio(
            normalized_a,
            normalized_b,
        )
        / 100.0
    )

    jaccard = jaccard_similarity(
        tokens_a,
        tokens_b,
    )

    containment = containment_similarity(
        tokens_a,
        tokens_b,
    )

    exact = (
        1.0
        if normalized_a == normalized_b
        else 0.0
    )

    return {
        "ratio": ratio,
        "jaccard": jaccard,
        "containment": containment,
        "exact": exact,
    }


# ============================================================
# ADDRESS FEATURES
# ============================================================

def calculate_address_features(address_a, address_b):
    """
    Calculate address similarity features.
    """

    normalized_a = normalize_address(
        address_a
    )

    normalized_b = normalize_address(
        address_b
    )

    tokens_a = address_tokens(
        address_a
    )

    tokens_b = address_tokens(
        address_b
    )

    signature_a = address_signature(
        address_a
    )

    signature_b = address_signature(
        address_b
    )

    signature_no_numbers_a = (
        address_signature_without_numbers(
            address_a
        )
    )

    signature_no_numbers_b = (
        address_signature_without_numbers(
            address_b
        )
    )

    if not normalized_a or not normalized_b:
        return {
            "ratio": 0.0,
            "jaccard": 0.0,
            "containment": 0.0,
            "signature": 0.0,
            "signature_no_numbers": 0.0,
        }

    ratio = (
        fuzz.ratio(
            normalized_a,
            normalized_b,
        )
        / 100.0
    )

    jaccard = jaccard_similarity(
        tokens_a,
        tokens_b,
    )

    containment = containment_similarity(
        tokens_a,
        tokens_b,
    )

    signature = (
        1.0
        if (
            signature_a
            and signature_b
            and signature_a == signature_b
        )
        else 0.0
    )

    signature_no_numbers = (
        1.0
        if (
            signature_no_numbers_a
            and signature_no_numbers_b
            and (
                signature_no_numbers_a
                == signature_no_numbers_b
            )
        )
        else 0.0
    )

    return {
        "ratio": ratio,
        "jaccard": jaccard,
        "containment": containment,
        "signature": signature,
        "signature_no_numbers":
            signature_no_numbers,
    }


# ============================================================
# MATCH SCORE
# ============================================================

def calculate_match_score(
    name,
    address,
    country_match,
):
    """
    Calculate the initial deterministic match score.

    Name is the strongest signal.
    Address provides supporting evidence.
    Country agreement is an additional signal.
    """

    name_score = max(
        name["exact"],
        (
            0.55 * name["ratio"]
            + 0.25 * name["jaccard"]
            + 0.20 * name["containment"]
        ),
    )

    address_score = max(
        address["signature"],
        (
            0.35 * address["ratio"]
            + 0.25 * address["jaccard"]
            + 0.20 * address["containment"]
            + 0.20 * address["signature_no_numbers"]
        ),
    )

    score = (
        0.60 * name_score
        + 0.30 * address_score
        + 0.10 * country_match
    )

    return score


# ============================================================
# SINGLE PAIR SCORING
# ============================================================

def score_pair(source1_record, candidate_record):
    """
    Score one Source-1 / candidate pair.
    """

    # Country mismatch is treated as a hard rejection.
    country_match = (
        1.0
        if same_country(
            source1_record["country"],
            candidate_record["country"],
        )
        else 0.0
    )

    if country_match == 0.0:
        return 0.0

    name = calculate_name_features(
        source1_record["business_name"],
        candidate_record["business_name"],
    )

    address = calculate_address_features(
        source1_record["business_address"],
        candidate_record["business_address"],
    )

    # Cheap evidence check.
    name_evidence = max(
        name["exact"],
        name["jaccard"],
        name["containment"],
        name["ratio"],
    )

    address_evidence = max(
        address["signature"],
        address["signature_no_numbers"],
        address["jaccard"],
        address["containment"],
        address["ratio"],
    )

    if (
        name_evidence < MIN_NAME_EVIDENCE
        and address_evidence < MIN_ADDRESS_EVIDENCE
    ):
        return 0.0

    return calculate_match_score(
        name,
        address,
        country_match,
    )


# ============================================================
# LOAD SOURCE
# ============================================================

def load_source(path):
    """
    Load only the columns required by matching.
    """

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )


# ============================================================
# BUILD LOOKUPS
# ============================================================

def build_lookup(df):
    """
    Build an entity_id -> record dictionary.

    This is done ONCE before candidate chunks are processed.
    """

    lookup = {}

    for row in df.itertuples(index=False):

        lookup[row.entity_id] = {
            "entity_id": row.entity_id,
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country,
        }

    return lookup


# ============================================================
# PROCESS ONE CANDIDATE CHUNK
# ============================================================

def process_candidate_chunk(
    candidates,
    source1_lookup,
    source2_lookup,
    source3_lookup,
    matches,
    threshold,
):
    """
    Process one candidate chunk.

    The matches dictionary is updated in-place.
    """

    processed = 0
    predicted = 0

    for row in candidates.itertuples(
        index=False
    ):

        source1_id = row.source1_entity_id

        source1_record = source1_lookup.get(
            source1_id
        )

        if source1_record is None:
            continue

        candidate_ids = parse_match_ids(
            row.candidate_entity_ids
        )

        for candidate_id in candidate_ids:

            # --------------------------------------------
            # Find candidate record
            # --------------------------------------------

            if candidate_id.startswith("S2-"):

                candidate_record = (
                    source2_lookup.get(
                        candidate_id
                    )
                )

            elif candidate_id.startswith("S3-"):

                candidate_record = (
                    source3_lookup.get(
                        candidate_id
                    )
                )

            else:
                continue

            if candidate_record is None:
                continue

            # --------------------------------------------
            # Calculate score
            # --------------------------------------------

            score = score_pair(
                source1_record,
                candidate_record,
            )

            # --------------------------------------------
            # MATCH
            # --------------------------------------------

            if score >= threshold:

                matches[
                    source1_id
                ].add(candidate_id)

                predicted += 1

        processed += 1

    return processed, predicted


# ============================================================
# CREATE FINAL RESULTS
# ============================================================

def create_results(
    source1,
    matches,
):
    """
    Create exactly one output row for every Source-1 entity.
    """

    rows = []

    for source1_id in source1["entity_id"]:

        matched_ids = sorted(
            matches.get(
                source1_id,
                set(),
            )
        )

        rows.append(
            {
                "source1_entity_id": source1_id,
                "matched_entity_ids":
                    ",".join(matched_ids),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Business Entity Resolution "
            "matching stage"
        )
    )

    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Dataset directory.",
    )

    parser.add_argument(
        "--candidates",
        default=(
            "output/"
            "candidate_pairs_v2a.tsv"
        ),
        help="V2-A candidate pairs.",
    )

    parser.add_argument(
        "--output",
        default=(
            "output/"
            "matching_results.tsv"
        ),
        help="Matching output.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Initial match threshold.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Candidate chunk size.",
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data_dir
    )

    # ========================================================
    # PATHS
    # ========================================================

    source1_path = (
        data_dir
        / "train"
        / "train_source1.tsv"
    )

    source2_path = (
        data_dir
        / "train"
        / "train_source2.tsv"
    )

    source3_path = (
        data_dir
        / "train"
        / "train_source3.tsv"
    )

    candidate_path = Path(
        args.candidates
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD SOURCES
    # ========================================================

    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION")
    print("MATCHING STAGE")
    print("=" * 60)

    print()
    print("Loading Source 1...")

    source1 = load_source(
        source1_path
    )

    print(
        f"Source 1 rows: "
        f"{len(source1):,}"
    )

    print("Loading Source 2...")

    source2 = load_source(
        source2_path
    )

    print(
        f"Source 2 rows: "
        f"{len(source2):,}"
    )

    print("Loading Source 3...")

    source3 = load_source(
        source3_path
    )

    print(
        f"Source 3 rows: "
        f"{len(source3):,}"
    )

    # ========================================================
    # BUILD LOOKUPS ONCE
    # ========================================================

    print()
    print("Building Source-1 lookup...")

    source1_lookup = build_lookup(
        source1
    )

    print("Building Source-2 lookup...")

    source2_lookup = build_lookup(
        source2
    )

    print("Building Source-3 lookup...")

    source3_lookup = build_lookup(
        source3
    )

    # ========================================================
    # INITIALIZE MATCH STORAGE
    # ========================================================

    matches = {
        source1_id: set()
        for source1_id in source1[
            "entity_id"
        ]
    }

    # ========================================================
    # PROCESS CANDIDATES IN CHUNKS
    # ========================================================

    print()
    print(
        f"Candidate file: "
        f"{candidate_path}"
    )

    print(
        f"Threshold: "
        f"{args.threshold}"
    )

    print(
        f"Chunk size: "
        f"{args.chunk_size:,}"
    )

    total_candidate_rows = 0
    total_predicted_matches = 0

    print()
    print("Starting matching...")

    for chunk_number, candidates in enumerate(
        pd.read_csv(
            candidate_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=args.chunk_size,
        ),
        start=1,
    ):

        processed, predicted = (
            process_candidate_chunk(
                candidates,
                source1_lookup,
                source2_lookup,
                source3_lookup,
                matches,
                args.threshold,
            )
        )

        total_candidate_rows += processed
        total_predicted_matches += predicted

        print(
            f"Chunk {chunk_number:,} | "
            f"candidate rows processed: "
            f"{total_candidate_rows:,} | "
            f"predicted matches: "
            f"{total_predicted_matches:,}"
        )

    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    results = create_results(
        source1,
        matches,
    )

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    entities_with_matches = sum(
        bool(value)
        for value in results[
            "matched_entity_ids"
        ]
    )

    print()
    print("=" * 60)
    print("MATCHING COMPLETE")
    print("=" * 60)

    print(
        f"Source-1 entities: "
        f"{len(results):,}"
    )

    print(
        f"Entities with matches: "
        f"{entities_with_matches:,}"
    )

    print(
        f"Total predicted matches: "
        f"{total_predicted_matches:,}"
    )

    print(
        f"Output: "
        f"{output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
