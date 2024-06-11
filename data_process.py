import os
import nibabel as nib
from PIL import Image
import torch
from skimage.measure import regionprops
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import numpy as np
from scipy.ndimage import binary_dilation
import copy
from tqdm import tqdm
import cv2

from utils import extract_mask_number
from config import Config


# from dl_config import Config





class MRIDataset_Unet_Classification(Dataset):
    def __init__(self, root_dir, transform=None, return_original=False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_original = return_original
        self.images, self.labels, self.sample_ids = self.load_images_and_labels()

    def load_images_and_labels(self):
        images = []
        labels = []
        sample_ids = []
        label_to_int = {}
        for label_dir in os.listdir(os.path.join(os.getcwd(), self.root_dir)):
            if label_dir not in ["Combination", "Sorafenib", "NK", "Control"]:
                continue
            class_path = os.path.join(self.root_dir, label_dir)
            if os.path.isdir(class_path):
                # binarization
                # label_dir = self.binarization(label_dir)
                label_int = label_to_int.setdefault(label_dir, len(label_to_int))
                for patient_id in os.listdir(class_path):
                    patient_path = os.path.join(class_path, patient_id)
                    for img_path in glob.glob(
                            os.path.join(class_path, patient_id, "*Separate*.nii.gz")
                    ):

                        mask_path = img_path.replace("Separate", "MASK")
                        mask_number = extract_mask_number(mask_path)
                        mask_path = os.path.join(
                            class_path, patient_id, "MASK_" + mask_number + ".nii.gz"
                        )
                        if os.path.isfile(img_path) and os.path.isfile(mask_path):
                            # check dimension mismatch
                            image = nib.load(img_path).get_fdata().astype(np.float32)
                            mask = nib.load(mask_path).get_fdata().astype(np.float32)
                            if image.shape != mask.shape:
                                continue
                            images.append((img_path, mask_path))
                            labels.append(label_int)
                            sample_ids.append(patient_id)
                # Config.num_classes = len(label_dir)

        return images, labels, sample_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, mask_path = self.images[idx]
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)

        # keep the original image for evaluation
        original_size = image.shape if self.return_original else None

        # image = torch.from_numpy(image)

        # if len(image.shape) == 2:  # for 2D images
        #     image = image.unsqueeze(0).repeat(3, 1, 1)

        # Convert image to 3-channel by repeating single channel
        # First, add an extra dimension at the beginning (axis=0)
        image = np.expand_dims(image, axis=2)
        # Then, repeat the array 3 times along the first dimension (axis=0)
        image = np.repeat(image, 3, axis=2)

        # Ensure image size correlate to mask size
        if image.shape[:2] != mask.shape:
            print(f"{img_path} has image shape: {image.shape}")
            print(f"{mask_path} has image shape: {mask.shape}")
        # Ensure masks are binary
        mask[mask > 0.0] = 1.0

        label = self.labels[idx]

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        if self.return_original:
            return image, mask, label, original_size[0], img_path
        else:
            return image, mask, label

    def binarization(self, label_dir):
        if label_dir in ["Combination", "Sorafenib", "NK"]:
            return "Treatment"
        else:
            return "Control"


class ZNormalization(torch.nn.Module):
    """
    Custom transform for performing z-normalization on an image.
    """

    def __init__(self):
        super(ZNormalization, self).__init__()

    def forward(self, img):
        """
        img: a PIL Image or a Torch tensor of shape (C, H, W)
        """
        if not isinstance(img, torch.Tensor):
            # Convert PIL Image to a tensor
            img = transforms.functional.to_tensor(img)

        mean = img.mean()
        std = img.std()
        if std == 0:
            std = 1
        normalized_img = (img - mean) / std
        return normalized_img


# # Define your transformations, e.g., Resize, ToTensor
# transform = transforms.Compose(
#     [
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         # transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
#         ZNormalization(),
#         # Add other transformations as needed
#     ]
# )

train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        #     max_pixel_value=255.0,
        # ),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # A.GaussNoise(var_limit=(10, 50), p=0.5),
        # ZNormalization(),
        ToTensorV2(),
    ],
    # is_check_shapes=False
)

val_transforms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

# transform = v2.Compose([
#     v2.ToPILImage(),
#     v2.RandomResizedCrop(size=(224, 224), antialias=True),
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# transform = transforms.Compose([
#     transforms.ToPILImage(),  # Convert to PIL Image first for most transformations
#     transforms.Resize((256, 256)),  # Resize the image to slightly larger than the final size
#     transforms.RandomCrop((224, 224)),  # Randomly crop to the final size
#     transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
#     transforms.RandomVerticalFlip(),  # Randomly flip the images vertically
#     transforms.RandomRotation(degrees=15),  # Randomly rotate the images in the range (-15, 15)
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly jitter the brightness, contrast, saturation and hue
#     transforms.ToTensor(),  # Convert the PIL Image to a tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with mean and std for pretrained models
# ])

# dataset = MRIDataset_Unet_Classification(Config.buffalo_dir, transform=train_transform, return_original=True)
# print("length of dataset", len(dataset))
# loader = DataLoader(dataset, batch_size=1, shuffle=True)
# for l in loader:
#     image, mask, label, original, original_size = l
#     print("after image size: ", image.size())
#     print("after mask size: ", mask.size())
#     print("label: ", label)
#     print("original: ", original.shape)
#     print("orignal size: ", original_size)
#     print("----------")
#     # break

# dataset = MRIDataset_Unet_Classification(Config.all_dir, transform=train_transform, return_original=True)
# print("length of dataset", len(dataset))
# loader = DataLoader(dataset, batch_size=8, shuffle=True)
# for i, (image, mask, label, original_size) in enumerate(tqdm(loader)):
#     # print("after image size: ", image.size())
#     # print("after mask size: ", mask.size())
#     # print("label: ", label)
#     # # print("original: ", original.shape)
#     # print("orignal size: ", original_size)
#     # print("----------")
#     # break

#
#
# dataset = MRIDataset_Unet(Config.buffalo_dir, transform=train_transform)
# print("length of dataset", len(dataset))
# loader = DataLoader(dataset, batch_size=1, shuffle=True)
# for l in loader:
#     image, mask = l
#     print("after image size: ", image.size())
#     print("after mask size: ", mask.size())
#     break
