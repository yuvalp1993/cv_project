"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        if index < len(self.real_image_names):
            # The index is for a real image
            img_name = self.real_image_names[index]
            label = 0
        else:
            # The index is for a fake image
            img_name = self.fake_image_names[index - len(self.real_image_names)]
            label = 1

        # Load the image
        img_path = os.path.join(self.root_path, 'real' if label == 0 else 'fake', img_name)
        image = Image.open(img_path)

        # Apply the transform if it's not None
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names)

