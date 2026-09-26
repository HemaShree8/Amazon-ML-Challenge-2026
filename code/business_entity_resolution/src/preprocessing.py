"""
Shared preprocessing and normalization utilities.

Used by:
    - candidate generation
    - matching
    - final integration

The original source fields should always be preserved.
These functions create normalized representations for
comparison and blocking.
"""

import re
import unicodedata


# ============================================================
# LEGAL SUFFIXES
# ============================================================

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "llc",
    "ltd",
    "limited",
    "llp",
    "lp",
    "plc",
    "pvt",
    "private",
    "pte",
    "gmbh",
    "sa",
    "sarl",
    "spa",
    "bv",
    "ag",
    "pc",
}


# ============================================================
# ADDRESS GENERIC TOKENS
# ============================================================

GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "lane",
    "ln",
    "drive",
    "dr",
    "highway",
    "hwy",
    "parkway",
    "pkwy",
    "way",
    "place",
    "pl",
    "court",
    "ct",
    "circle",
    "cir",
    "square",
    "sq",
    "building",
    "bldg",
    "floor",
    "fl",
    "suite",
    "ste",
    "unit",
    "apartment",
    "apt",
    "block",
    "sector",
    "phase",
    "district",
    "city",
    "state",
    "country",
}


# ============================================================
# COMMON ABBREVIATIONS
# ============================================================

# These are intentionally conservative.
# Do not aggressively replace arbitrary words.

NAME_ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "pkwy": "parkway",
    "ct": "court",
    "cir": "circle",
    "sq": "square",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "apt": "apartment",
}


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_unicode(value):
    """
    Normalize Unicode using NFKC.

    This keeps different scripts intact while normalizing
    compatible Unicode representations.
    """

    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return unicodedata.normalize(
        "NFKC",
        text,
    )


def normalize_text(value):
    """
    General text normalization.

    Steps:
        1. Missing value -> ""
        2. Unicode NFKC normalization
        3. Unicode casefold
        4. Normalize apostrophes
        5. Replace punctuation with spaces
        6. Collapse whitespace

    Example:

        "  ABC-Tech, Inc.  "
            -> "abc tech inc"
    """

    text = normalize_unicode(value)

    if not text:
        return ""

    text = text.casefold()

    # Normalize apostrophe-like characters.
    text = re.sub(
        r"['’‘`´]",
        "",
        text,
    )

    # Replace punctuation/symbols with spaces.
    #
    # \w preserves Unicode letters/numbers.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_name(value):
    """
    Normalize a business name.

    Important:
        Apostrophes are removed rather than replaced by spaces.

    Therefore:

        McDonald's
            -> mcdonalds

        O'Reilly
            -> oreilly

    Common legal suffixes are removed.

    Examples:

        "McDonald's Pvt. Ltd."
            -> "mcdonalds"

        "McDonalds Pvt Ltd"
            -> "mcdonalds"

        "MCDONALDS, PVT. LTD"
            -> "mcdonalds"

        "ABC-Tech, Inc."
            -> "abc tech"
    """

    text = normalize_text(value)

    if not text:
        return ""

    tokens = text.split()

    cleaned = []

    for token in tokens:

        # Remove legal suffixes.
        if token in LEGAL_SUFFIXES:
            continue

        # Ignore one-character alphabetic noise.
        if len(token) == 1 and not token.isdigit():
            continue

        cleaned.append(token)

    return " ".join(cleaned)


def name_tokens(value):
    """
    Return meaningful normalized business-name tokens.
    """

    normalized = normalize_name(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


def normalize_name_for_blocking(value):
    """
    Alias intended for candidate-generation code.

    Keeping this separate makes the intent explicit and lets
    candidate generation import a stable function.
    """

    return normalize_name(value)


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(value):
    """
    Normalize an address.

    Unlike business names, address numbers and components
    are preserved.

    Examples:

        "12, M.G. Road, Bengaluru"
            -> "12 m g road bengaluru"

        "12 MG Rd Bengaluru"
            -> "12 mg rd bengaluru"
    """

    text = normalize_text(value)

    if not text:
        return ""

    return text


def address_tokens(value):
    """
    Return useful address tokens.

    Generic address words are removed because they are usually
    weaker matching signals.

    Numbers and alphanumeric components are retained.

    Example:

        "12 MG Road, Bangalore"
            -> {"12", "mg", "bangalore"}
    """

    normalized = normalize_address(value)

    if not normalized:
        return set()

    tokens = normalized.split()

    result = set()

    for token in tokens:

        # Remove generic address terms.
        if token in GENERIC_ADDRESS_TOKENS:
            continue

        # Ignore one-character alphabetic noise.
        if len(token) == 1 and not token.isdigit():
            continue

        result.add(token)

    return result


# ============================================================
# ADDRESS SIGNATURES
# ============================================================

def address_signature(value):
    """
    Order-independent address signature.

    Example:

        "123 Main Street Bangalore"
        "Bangalore 123 Main Street"

    produce the same signature.
    """

    tokens = address_tokens(value)

    if not tokens:
        return ""

    return " ".join(
        sorted(tokens)
    )


def address_signature_without_numbers(value):
    """
    Order-independent address signature with pure numeric
    tokens removed.

    Useful when one source contains a house number and
    another source omits it.
    """

    tokens = address_tokens(value)

    tokens = {
        token
        for token in tokens
        if not token.isdigit()
    }

    if not tokens:
        return ""

    return " ".join(
        sorted(tokens)
    )


# ============================================================
# TOKEN SIMILARITY HELPERS
# ============================================================

def jaccard_similarity(tokens_a, tokens_b):
    """
    Jaccard similarity between two token sets.
    """

    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(
        tokens_a & tokens_b
    ) / len(union)


def containment_similarity(tokens_a, tokens_b):
    """
    Fraction of the smaller token set contained in the larger.
    """

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(
        tokens_a & tokens_b
    )

    smaller = min(
        len(tokens_a),
        len(tokens_b),
    )

    if smaller == 0:
        return 0.0

    return intersection / smaller


# ============================================================
# RECORD PREPROCESSING
# ============================================================

def preprocess_record(
    business_name,
    business_address,
):
    """
    Preprocess one business record.

    Returns all commonly required normalized representations.
    """

    return {
        "normalized_name":
            normalize_name(
                business_name
            ),

        "name_tokens":
            name_tokens(
                business_name
            ),

        "normalized_address":
            normalize_address(
                business_address
            ),

        "address_tokens":
            address_tokens(
                business_address
            ),

        "address_signature":
            address_signature(
                business_address
            ),

        "address_signature_without_numbers":
            address_signature_without_numbers(
                business_address
            ),
    }