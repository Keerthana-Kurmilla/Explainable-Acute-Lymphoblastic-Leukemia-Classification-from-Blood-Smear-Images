from torchvision import transforms
import numpy as np
from PIL import Image
from skimage import color


# ImageNet normalization for CLIP-style models
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# Training transformation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CLIP_MEAN,
        std=CLIP_STD
    )
])


# Validation / Test transformation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CLIP_MEAN,
        std=CLIP_STD
    )
])


# Reinhard stain normalization
def reinhard_normalization(image, reference_image):

    image = image.convert("RGB")
    reference_image = reference_image.convert("RGB")

    # Convert images to NumPy
    image_np = np.asarray(image).astype(np.float32) / 255.0
    reference_np = np.asarray(reference_image).astype(np.float32) / 255.0

    # RGB → LAB
    lab = color.rgb2lab(image_np)
    reference_lab = color.rgb2lab(reference_np)

    # Create foreground masks
    image_gray = np.mean(image_np, axis=2)
    reference_gray = np.mean(reference_np, axis=2)

    image_mask = image_gray > 0.05
    reference_mask = reference_gray > 0.05

    # Get LAB pixels only from the cell/foreground
    source_pixels = lab[image_mask]
    target_pixels = reference_lab[reference_mask]

    # Calculate source statistics
    source_mean = source_pixels.mean(axis=0)
    source_std = source_pixels.std(axis=0)

    # Calculate target statistics from reference image
    target_mean = target_pixels.mean(axis=0)
    target_std = target_pixels.std(axis=0)

    # Avoid division by zero
    source_std = np.maximum(source_std, 1e-6)

    # Reinhard normalization
    normalized_lab = (
        (lab - source_mean)
        / source_std
        * target_std
        + target_mean
    )

    # Convert LAB → RGB
    normalized_rgb = color.lab2rgb(normalized_lab)

    # Clip values
    normalized_rgb = np.clip(normalized_rgb, 0, 1)

    # Keep background black
    normalized_rgb[~image_mask] = 0

    # Convert to uint8
    normalized_rgb = (
        normalized_rgb * 255
    ).astype(np.uint8)

    return Image.fromarray(normalized_rgb)