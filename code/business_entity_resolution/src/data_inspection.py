import pandas as pd

source1_path = "dataset/train/train_source1.tsv"

source1 = pd.read_csv(source1_path, sep="\t")

print(source1.head())
print(source1.shape)
print(source1.dtypes)

print("check missing values that exist in each column")
print(source1.isna().sum())

print("how the 2.2 million Source-1 businesses are distributed across countries?")
print(source1["country"].value_counts())

print("whether Source 1 has duplicate entity_ids.")
print("Duplicate entity IDs:", source1["entity_id"].duplicated().sum())

print("Business name duplications")
print("Duplicate business names:", source1["business_name"].duplicated().sum())

print("how often the exact address repeats in Source 1.")
print("Duplicate business addresses:", source1["business_address"].duplicated().sum())

source2_path = "dataset/train/train_source2.tsv"

source2 = pd.read_csv(source2_path, sep="\t")

print("\nSOURCE 2")
print(source2.head())
print(source2.shape)
print(source2.dtypes)
print(source2.isna().sum())
print(source2["country"].value_counts())