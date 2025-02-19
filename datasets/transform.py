from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomAdjustSharpness,
    RandomRotation,
    Resize,
)


def train_transform() -> Compose:
    """
    Returns a composition of transforms to be applied to the training images.

    Returns:
        Compose: A composition of transforms.
    """
    return Compose(
        [
            RandomRotation(degrees=10),
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            RandomAdjustSharpness(sharpness_factor=1.2, p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def test_transform() -> Compose:
    """
    Creates a composition of image transformations for test and validation.

    Returns:
        Compose: A composition of the specified image transformations.
    """
    return Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
