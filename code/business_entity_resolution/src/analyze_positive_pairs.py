import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio


def normalize_text(value):
    """
    Lightweight normalization for analysis only.
    Do NOT treat this as the final normalization pipeline.
    """
    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase / casefold
    value = value.casefold()

    # Replace punctuation with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def load_source(path, columns):
    print(f"Loading {path}...")
    return pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype=str,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset directory"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of positive S1 entities to sample"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_dir = data_dir / "train"

    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # ---------------------------------------------------------
    # 1. Load ground truth
    # ---------------------------------------------------------

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
    )

    # Only entities having at least one match
    gt_positive = gt[
        gt["matched_entity_ids"].fillna("").str.strip() != ""
    ].copy()

    print(f"Total S1 entities: {len(gt):,}")
    print(f"S1 entities with matches: {len(gt_positive):,}")

    # Sample positive S1 entities
    sample_n = min(args.sample_size, len(gt_positive))

    sampled_gt = gt_positive.sample(
        n=sample_n,
        random_state=42
    ).copy()

    print(f"Sampled positive S1 entities: {sample_n:,}")

    # ---------------------------------------------------------
    # 2. Convert comma-separated matches into rows
    # ---------------------------------------------------------

    pairs = sampled_gt[
        ["source1_entity_id", "matched_entity_ids"]
    ].copy()

    pairs["matched_entity_ids"] = pairs["matched_entity_ids"].str.split(",")

    pairs = pairs.explode("matched_entity_ids")

    pairs["matched_entity_ids"] = (
        pairs["matched_entity_ids"]
        .astype(str)
        .str.strip()
    )

    pairs = pairs[
        pairs["matched_entity_ids"] != ""
    ]

    print(f"Positive pairs sampled: {len(pairs):,}")

    # ---------------------------------------------------------
    # 3. Load Source 1
    # ---------------------------------------------------------

    s1 = load_source(
        s1_path,
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s1 = s1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "s1_name",
            "business_address": "s1_address",
            "country": "s1_country",
        }
    )

    # ---------------------------------------------------------
    # 4. Load Source 2
    # ---------------------------------------------------------

    s2 = load_source(
        s2_path,
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s2 = s2.rename(
        columns={
            "entity_id": "matched_entity_ids",
            "business_name": "s2_name",
            "business_address": "s2_address",
            "country": "s2_country",
        }
    )

    # ---------------------------------------------------------
    # 5. Load Source 3
    # ---------------------------------------------------------

    s3 = load_source(
        s3_path,
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s3 = s3.rename(
        columns={
            "entity_id": "matched_entity_ids",
            "business_name": "s3_name",
            "business_address": "s3_address",
            "country": "s3_country",
        }
    )

    # ---------------------------------------------------------
    # 6. Determine whether matched ID belongs to S2 or S3
    # ---------------------------------------------------------

    s2_ids = set(s2["matched_entity_ids"])
    s3_ids = set(s3["matched_entity_ids"])

    pairs["source"] = pairs["matched_entity_ids"].map(
        lambda x: "S2" if x in s2_ids else ("S3" if x in s3_ids else "UNKNOWN")
    )

    print("\nSource distribution:")
    print(pairs["source"].value_counts())

    # ---------------------------------------------------------
    # 7. Join S1 records
    # ---------------------------------------------------------

    pairs = pairs.merge(
        s1,
        on="source1_entity_id",
        how="left",
    )

    # ---------------------------------------------------------
    # 8. Join matching S2/S3 record
    # ---------------------------------------------------------

    s2_pairs = pairs[pairs["source"] == "S2"].copy()
    s3_pairs = pairs[pairs["source"] == "S3"].copy()

    # S2
    if len(s2_pairs) > 0:
        s2_pairs = s2_pairs.merge(
            s2,
            on="matched_entity_ids",
            how="left",
        )

        s2_pairs["matched_name"] = s2_pairs["s2_name"]
        s2_pairs["matched_address"] = s2_pairs["s2_address"]
        s2_pairs["matched_country"] = s2_pairs["s2_country"]

    # S3
    if len(s3_pairs) > 0:
        s3_pairs = s3_pairs.merge(
            s3,
            on="matched_entity_ids",
            how="left",
        )

        s3_pairs["matched_name"] = s3_pairs["s3_name"]
        s3_pairs["matched_address"] = s3_pairs["s3_address"]
        s3_pairs["matched_country"] = s3_pairs["s3_country"]

    # Combine
    result_parts = []

    if len(s2_pairs) > 0:
        result_parts.append(s2_pairs)

    if len(s3_pairs) > 0:
        result_parts.append(s3_pairs)

    result = pd.concat(
        result_parts,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # 9. Calculate similarities
    # ---------------------------------------------------------

    result["s1_name_norm"] = result["s1_name"].map(normalize_text)
    result["matched_name_norm"] = result["matched_name"].map(normalize_text)

    result["s1_address_norm"] = result["s1_address"].map(normalize_text)
    result["matched_address_norm"] = result["matched_address"].map(normalize_text)

    result["same_country"] = (
        result["s1_country"].fillna("")
        == result["matched_country"].fillna("")
    )

    result["exact_name_raw"] = (
        result["s1_name"].fillna("")
        == result["matched_name"].fillna("")
    )

    result["exact_name_normalized"] = (
        result["s1_name_norm"]
        == result["matched_name_norm"]
    )

    result["exact_address_raw"] = (
        (result["s1_address"].notna())
        & (result["matched_address"].notna())
        & (
            result["s1_address"]
            == result["matched_address"]
        )
    )

    result["exact_address_normalized"] = (
        (result["s1_address_norm"] != "")
        & (result["matched_address_norm"] != "")
        & (
            result["s1_address_norm"]
            == result["matched_address_norm"]
        )
    )

    # Fuzzy similarities
    result["name_similarity"] = result.apply(
        lambda row: ratio(
            row["s1_name_norm"],
            row["matched_name_norm"]
        ) / 100
        if row["s1_name_norm"] and row["matched_name_norm"]
        else 0,
        axis=1,
    )

    result["address_similarity"] = result.apply(
        lambda row: ratio(
            row["s1_address_norm"],
            row["matched_address_norm"]
        ) / 100
        if row["s1_address_norm"] and row["matched_address_norm"]
        else 0,
        axis=1,
    )

    # ---------------------------------------------------------
    # 10. Print statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("POSITIVE PAIR ANALYSIS")
    print("=" * 70)

    print("\nCountry agreement:")
    print(
        result["same_country"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nRaw exact name:")
    print(
        result["exact_name_raw"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nNormalized exact name:")
    print(
        result["exact_name_normalized"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nRaw exact address:")
    print(
        result["exact_address_raw"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nNormalized exact address:")
    print(
        result["exact_address_normalized"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nName similarity:")
    print(
        result["name_similarity"].describe()
    )

    print("\nAddress similarity:")
    print(
        result["address_similarity"].describe()
    )

    # ---------------------------------------------------------
    # 11. Show difficult examples
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DIFFICULT POSITIVE PAIRS")
    print("=" * 70)

    difficult = result[
        (~result["exact_name_normalized"])
        | (~result["exact_address_normalized"])
    ].copy()

    difficult = difficult.sort_values(
        [
            "name_similarity",
            "address_similarity",
        ]
    )

    columns_to_show = [
        "source",
        "source1_entity_id",
        "matched_entity_ids",
        "s1_country",
        "s1_name",
        "matched_name",
        "s1_address",
        "matched_address",
        "name_similarity",
        "address_similarity",
        "exact_name_normalized",
        "exact_address_normalized",
    ]

    print(
        difficult[
            columns_to_show
        ].head(30).to_string(index=False)
    )

    # ---------------------------------------------------------
    # 12. Save analysis
    # ---------------------------------------------------------

    output_path = Path("positive_pair_analysis.tsv")

    result[
        columns_to_show
    ].to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved detailed results to: {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()