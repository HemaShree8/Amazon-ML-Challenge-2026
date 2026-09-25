"""
Candidate generation / blocking for Business Entity Resolution.

Goal:
    Reduce the number of S1 -> S2/S3 comparisons before
    similarity calculation and matching.

This is a first blocking version.

Blocking rules:
    1. Exact normalized business name + country
    2. First meaningful name token + country
    3. First two meaningful name tokens + country

The rules are combined so that a candidate found by ANY rule
is retained.

Output:
    output/candidate_pairs.tsv
"""

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Common legal/business suffixes.
# These are removed ONLY for blocking purposes.
LEGAL_SUFFIXES = {
    "pvt",
    "private",
    "ltd",
    "limited",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
}


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(value):
    """
    Basic text normalization.

    Example:

        "ABC Technologies, Pvt. Ltd."
        ->
        "abc technologies pvt ltd"
    """

    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.casefold()

    # Replace punctuation with spaces
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Collapse whitespace
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def normalize_name(value):
    """
    Normalize business name for blocking.

    Legal suffixes are removed so that:

        ABC Technologies Pvt Ltd
        ABC Technologies Private Limited

    can enter the same block.
    """

    text = normalize_text(value)

    if not text:
        return ""

    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(tokens)


def normalize_country(value):
    """
    Normalize country for blocking.
    """

    return normalize_text(value)


# ---------------------------------------------------------
# BLOCKING KEY GENERATION
# ---------------------------------------------------------

def get_name_tokens(normalized_name):
    """
    Return meaningful tokens from a normalized name.

    Very short tokens are ignored for blocking because
    they create extremely large blocks.
    """

    if not normalized_name:
        return []

    tokens = normalized_name.split()

    # Ignore extremely short tokens.
    tokens = [
        token
        for token in tokens
        if len(token) >= 2
    ]

    return tokens


def generate_block_keys(name, country):
    """
    Generate multiple blocking keys for one business.

    Returns a dictionary containing:

        exact
        first_token
        first_two_tokens
    """

    normalized_name = normalize_name(name)
    normalized_country = normalize_country(country)

    if not normalized_name or not normalized_country:
        return {
            "exact": None,
            "first_token": None,
            "first_two_tokens": None,
        }

    tokens = get_name_tokens(normalized_name)

    if not tokens:
        return {
            "exact": None,
            "first_token": None,
            "first_two_tokens": None,
        }

    keys = {
        "exact": (
            normalized_country,
            normalized_name
        ),

        "first_token": (
            normalized_country,
            tokens[0]
        ),

        "first_two_tokens": (
            normalized_country,
            " ".join(tokens[:2])
        ),
    }

    return keys


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_source(path):
    """
    Load only the columns required for candidate generation.
    """

    print(f"Loading: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    return df


# ---------------------------------------------------------
# BUILD BLOCK INDEX
# ---------------------------------------------------------

def build_block_index(df, source_name):
    """
    Build indexes for a Source-2 or Source-3 dataframe.

    Each index maps a blocking key to a list of entity IDs.
    """

    exact_index = defaultdict(list)
    first_token_index = defaultdict(list)
    first_two_index = defaultdict(list)

    print(
        f"Building blocking indexes for {source_name}..."
    )

    for row in df.itertuples(index=False):

        entity_id = row.entity_id
        name = row.business_name
        country = row.country

        keys = generate_block_keys(
            name,
            country
        )

        if keys["exact"] is not None:
            exact_index[keys["exact"]].append(
                entity_id
            )

        if keys["first_token"] is not None:
            first_token_index[keys["first_token"]].append(
                entity_id
            )

        if keys["first_two_tokens"] is not None:
            first_two_index[keys["first_two_tokens"]].append(
                entity_id
            )

    print(
        f"{source_name} exact blocks: "
        f"{len(exact_index):,}"
    )

    print(
        f"{source_name} first-token blocks: "
        f"{len(first_token_index):,}"
    )

    print(
        f"{source_name} first-two-token blocks: "
        f"{len(first_two_index):,}"
    )

    return {
        "exact": exact_index,
        "first_token": first_token_index,
        "first_two_tokens": first_two_index,
    }


# ---------------------------------------------------------
# CANDIDATE GENERATION FOR ONE RECORD
# ---------------------------------------------------------

def generate_candidates_for_record(
    name,
    country,
    indexes,
    max_first_token_candidates=200,
):
    """
    Generate candidate entity IDs for one Source-1 record.

    Candidates are obtained from multiple blocking rules.

    A safety limit is applied to very large first-token blocks.
    """

    keys = generate_block_keys(
        name,
        country
    )

    candidates = set()

    # -----------------------------------------------------
    # Rule 1: exact normalized name + country
    # -----------------------------------------------------

    exact_key = keys["exact"]

    if exact_key is not None:

        candidates.update(
            indexes["exact"].get(
                exact_key,
                []
            )
        )

    # -----------------------------------------------------
    # Rule 2: first two tokens + country
    # -----------------------------------------------------

    first_two_key = keys["first_two_tokens"]

    if first_two_key is not None:

        candidates.update(
            indexes["first_two_tokens"].get(
                first_two_key,
                []
            )
        )

    # -----------------------------------------------------
    # Rule 3: first token + country
    # -----------------------------------------------------

    first_token_key = keys["first_token"]

    if first_token_key is not None:

        block = indexes["first_token"].get(
            first_token_key,
            []
        )

        # Very common first tokens can produce enormous
        # candidate sets. Limit them in V1.
        if len(block) <= max_first_token_candidates:

            candidates.update(block)

    return candidates


# ---------------------------------------------------------
# GENERATE CANDIDATES FOR SOURCE 1
# ---------------------------------------------------------

def generate_candidates(
    source1,
    source2,
    source3,
    max_first_token_candidates=200,
):
    """
    Generate candidate S2/S3 IDs for every S1 entity.
    """

    print("\nBuilding Source-2 indexes...")

    s2_indexes = build_block_index(
        source2,
        "Source 2"
    )

    print("\nBuilding Source-3 indexes...")

    s3_indexes = build_block_index(
        source3,
        "Source 3"
    )

    results = []

    print("\nGenerating candidates...")

    total = len(source1)

    for counter, row in enumerate(
        source1.itertuples(index=False),
        start=1
    ):

        candidates = set()

        # ---------------------------------------------
        # Source 2 candidates
        # ---------------------------------------------

        candidates.update(
            generate_candidates_for_record(
                row.business_name,
                row.country,
                s2_indexes,
                max_first_token_candidates
            )
        )

        # ---------------------------------------------
        # Source 3 candidates
        # ---------------------------------------------

        candidates.update(
            generate_candidates_for_record(
                row.business_name,
                row.country,
                s3_indexes,
                max_first_token_candidates
            )
        )

        # ---------------------------------------------
        # Store result
        # ---------------------------------------------

        results.append({
            "source1_entity_id": row.entity_id,
            "candidate_entity_ids": ",".join(
                sorted(candidates)
            ),
        })

        # Progress
        if counter % 100_000 == 0:

            print(
                f"Processed "
                f"{counter:,}/{total:,} "
                f"S1 records"
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

def print_candidate_statistics(candidates):
    """
    Print basic candidate-generation statistics.
    """

    candidate_counts = (
        candidates["candidate_entity_ids"]
        .fillna("")
        .apply(
            lambda x:
            0 if not x else len(x.split(","))
        )
    )

    print("\n" + "=" * 60)
    print("CANDIDATE GENERATION STATISTICS")
    print("=" * 60)

    print(
        "Source-1 entities:",
        f"{len(candidates):,}"
    )

    print(
        "Entities with at least one candidate:",
        f"{(candidate_counts > 0).sum():,}"
    )

    print(
        "Entities with zero candidates:",
        f"{(candidate_counts == 0).sum():,}"
    )

    print(
        "\nCandidate count statistics:"
    )

    print(
        candidate_counts.describe()
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate pairs for "
            "business entity resolution."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset directory"
    )

    parser.add_argument(
        "--output",
        default="output/candidate_pairs.tsv",
        help="Output candidate TSV path"
    )

    parser.add_argument(
        "--max-first-token-candidates",
        type=int,
        default=200,
        help=(
            "Maximum size of a first-token block "
            "used for candidate generation."
        )
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_dir = data_dir / "train"

    # -----------------------------------------------------
    # Load training sources
    # -----------------------------------------------------

    source1_path = train_dir / "train_source1.tsv"
    source2_path = train_dir / "train_source2.tsv"
    source3_path = train_dir / "train_source3.tsv"

    source1 = load_source(source1_path)
    source2 = load_source(source2_path)
    source3 = load_source(source3_path)

    print("\nDataset sizes:")
    print(
        "Source 1:",
        f"{len(source1):,}"
    )
    print(
        "Source 2:",
        f"{len(source2):,}"
    )
    print(
        "Source 3:",
        f"{len(source3):,}"
    )

    # -----------------------------------------------------
    # Generate candidates
    # -----------------------------------------------------

    candidates = generate_candidates(
        source1,
        source2,
        source3,
        max_first_token_candidates=(
            args.max_first_token_candidates
        ),
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    print_candidate_statistics(
        candidates
    )

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(
        f"\nCandidate pairs saved to:"
        f"\n{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
