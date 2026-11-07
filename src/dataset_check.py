from pathlib import Path

# CHANGE THIS PATH to your actual C-NMC_training_data location
DATASET_PATH = Path(r"C:\Users\DELL\Downloads\C-NMC 2019 (PKG)\C-NMC_training_data")

print("Checking C-NMC dataset...")
print()

for fold in ["fold_0", "fold_1", "fold_2"]:
    fold_path = DATASET_PATH / fold

    all_path = fold_path / "all"
    hem_path = fold_path / "hem"

    all_images = list(all_path.glob("*.bmp"))
    hem_images = list(hem_path.glob("*.bmp"))

    print(f"{fold}:")
    print(f"  ALL images : {len(all_images)}")
    print(f"  HEM images : {len(hem_images)}")
    print(f"  Total      : {len(all_images) + len(hem_images)}")
    print()