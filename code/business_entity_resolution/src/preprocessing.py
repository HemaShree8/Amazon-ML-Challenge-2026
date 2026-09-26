import re
import unicodedata


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

import pandas as pd

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=10
)

for name in source1["business_name"]:
    print("Original :", name)
    print("Normalized:", normalize_text(name))
    print()

import pandas as pd

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=10
)

print("BUSINESS NAMES")
print("=" * 50)

for name in source1["business_name"]:
    print("Original :", name)
    print("Normalized:", normalize_text(name))
    print()


print("BUSINESS ADDRESSES")
print("=" * 50)

for address in source1["business_address"]:
    print("Original :", address)
    print("Normalized:", normalize_text(address))
    print()