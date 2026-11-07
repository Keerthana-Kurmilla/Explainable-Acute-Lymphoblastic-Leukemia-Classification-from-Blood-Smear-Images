from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# CHANGE THIS TO YOUR ACTUAL C-NMC TRAINING PATH
# --------------------------------------------------

DATASET_PATH = Path(
    r"C:\Users\DELL\Downloads\C-NMC 2019 (PKG)\C-NMC_training_data"
)

OUTPUT_PATH = Path("data/manifests")

# Create output directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# COLLECT IMAGES FROM THE THREE FOLDS
# --------------------------------------------------

records = []

for fold in ["fold_0", "fold_1", "fold_2"]:

    fold_path = DATASET_PATH / fold

    # ALL = Leukemia
    all_path = fold_path / "all"

    # HEM = Normal
    hem_path = fold_path / "hem"

    # ALL images
    for image_path in all_path.glob("*.bmp"):
        records.append({
            "path": str(image_path.resolve()),
            "label": 1,
            "class": "ALL",
            "fold": fold
        })

    # HEM images
    for image_path in hem_path.glob("*.bmp"):
        records.append({
            "path": str(image_path.resolve()),
            "label": 0,
            "class": "HEM",
            "fold": fold
        })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)

print("Total images:", len(df))
print()
print("Class distribution:")
print(df["class"].value_counts())
print()
print("Fold distribution:")
print(df["fold"].value_counts())


# --------------------------------------------------
# DEVELOPMENT DATA = FOLD 0 + FOLD 1
# TEST DATA = FOLD 2
# --------------------------------------------------

development_df = df[df["fold"].isin(["fold_0", "fold_1"])].copy()

test_df = df[df["fold"] == "fold_2"].copy()


# --------------------------------------------------
# SPLIT DEVELOPMENT DATA INTO TRAIN + VALIDATION
# --------------------------------------------------

train_df, val_df = train_test_split(
    development_df,
    test_size=0.20,
    random_state=42,
    stratify=development_df["label"]
)


# --------------------------------------------------
# SAVE CSV FILES
# --------------------------------------------------

train_df.to_csv(
    OUTPUT_PATH / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_PATH / "val.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_PATH / "test.csv",
    index=False
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print()
print("Dataset split completed!")
print()

print("TRAIN:")
print(train_df["class"].value_counts())

print()
print("VALIDATION:")
print(val_df["class"].value_counts())

print()
print("TEST:")
print(test_df["class"].value_counts())

print()
print("Files created:")
print("data/manifests/train.csv")
print("data/manifests/val.csv")
print("data/manifests/test.csv")