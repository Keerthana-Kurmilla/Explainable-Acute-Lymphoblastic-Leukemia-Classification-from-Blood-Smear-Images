import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CNMCDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None,
        reinhard_fn=None,
        reference_image=None
    ):

        self.data = pd.read_csv(csv_file)

        self.transform = transform
        self.reinhard_fn = reinhard_fn
        self.reference_image = reference_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["path"]
        label = int(row["label"])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Reinhard normalization
        if (
            self.reinhard_fn is not None
            and self.reference_image is not None
        ):
            image = self.reinhard_fn(
                image,
                self.reference_image
            )

        # Apply resize / augmentation / tensor conversion
        if self.transform is not None:
            image = self.transform(image)

        return image, label