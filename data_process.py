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
    """
    Dataset for single MRI image
    """
    def __init__(self, root_dir, transform=None, return_original=False, preprocessing=None):
        self.root_dir = root_dir
        self.transform = transform
        self.return_original = return_original
        self.preprocessing = preprocessing
        self.images, self.labels, self.sample_ids, self.label_to_int = self.load_images_and_labels()

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
                        if mask_number is None:
                            print(f"Warning: Could not extract mask number from path: {mask_path}")
                            continue
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
        print("label to int", label_to_int)
        return images, labels, sample_ids, label_to_int

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

        # Dilate the mask to increase the ROI size
        # structure = np.ones((3, 3))  # Define the structuring element for dilation
        # mask = binary_dilation(mask, structure=structure).astype(np.float32)

        label = self.labels[idx]

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.return_original:
            return image, mask, label, original_size[0], img_path
        else:
            return image, mask, label

    def binarization(self, label_dir):
        if label_dir in ["Combination", "Sorafenib", "NK"]:
            return "Treatment"
        else:
            return "Control"


class MultiSequenceMRIDataset(Dataset):
    """
    Dataset for multi-sequence MRI images (put in different channels)
    """
    def __init__(self, root_dir, transform=None, return_original=False, preprocessing=None):
        self.root_dir = root_dir
        self.transform = transform
        self.return_original = return_original
        self.preprocessing = preprocessing
        self.images, self.labels, self.sample_ids, self.label_to_int = self._load_images_and_labels()

    def _load_images_and_labels(self):
        images = []
        labels = []
        sample_ids = []
        label_to_int = {}

        for label_dir in self._get_label_dirs():
            class_path = os.path.join(self.root_dir, label_dir)
            label_int = label_to_int.setdefault(label_dir, len(label_to_int))

            for patient_id in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_id)

                t1w_paths = glob.glob(os.path.join(patient_path, "T1W_HR_Separate_*.nii.gz"))
                t2w_paths = glob.glob(os.path.join(patient_path, "T2W_HR_Separate_*.nii.gz"))
                t1w_pc_paths = glob.glob(os.path.join(patient_path, "T1W_PC_Separate_*.nii.gz"))

                if not t1w_paths or not t2w_paths or not t1w_pc_paths:
                    print("No data path for {}".format(patient_path))
                    continue

                for t1w_path, t2w_path, t1w_pc_path in zip(t1w_paths, t2w_paths, t1w_pc_paths):
                    mask_path = self._get_mask_path(t1w_path)
                    if not os.path.isfile(mask_path):
                        continue

                    if self._validate_image_mask_shapes([t1w_path, t2w_path, t1w_pc_path, mask_path]):
                        images.append((t1w_path, t2w_path, t1w_pc_path, mask_path))
                        labels.append(label_int)
                        sample_ids.append(patient_id)

        print("label to int", label_to_int)
        return images, labels, sample_ids, label_to_int

    def _get_label_dirs(self):
        return [d for d in os.listdir(os.path.join(os.getcwd(), self.root_dir)) if d in ["Combination", "Sorafenib", "NK", "Control"]]

    def _get_mask_path(self, t1w_path):
        mask_path = t1w_path.replace("Separate", "MASK")
        mask_number = extract_mask_number(mask_path)
        if mask_number is None:
            print(f"Warning: Could not extract mask number from path: {mask_path}")
            return
        parent_dir = os.path.dirname(t1w_path)
        mask_path = os.path.join(parent_dir, "MASK_" + mask_number + ".nii.gz")
        return mask_path


    def _validate_image_mask_shapes(self, paths):
        images = [nib.load(p).get_fdata().astype(np.float32) for p in paths[:3]]
        mask = nib.load(paths[3]).get_fdata().astype(np.float32)
        return all(image.shape == mask.shape for image in images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        t1w_path, t2w_path, t1w_pc_path, mask_path = self.images[idx]
        t1w_image, t2w_image, t1w_pc_image, mask = self._load_images(t1w_path, t2w_path, t1w_pc_path, mask_path)

        image = self._merge_images(t1w_image, t2w_image, t1w_pc_image)
        original_size = image.shape if self.return_original else None

        mask = self._process_mask(mask)
        label = self.labels[idx]

        if self.transform:
            image, mask = self._apply_transform(image, mask)
        if self.preprocessing:
            image, mask = self._apply_preprocessing(image, mask)

        if self.return_original:
            return image, mask, label, original_size[0], t1w_path
        else:
            return image, mask, label

    def _load_images(self, *paths):
        return [nib.load(p).get_fdata().astype(np.float32) for p in paths]

    def _merge_images(self, t1w_image, t2w_image, t1w_pc_image):
        return np.stack((t1w_image, t2w_image, t1w_pc_image), axis=2)

    def _process_mask(self, mask):
        mask[mask > 0.0] = 1.0
        return mask

    def _apply_transform(self, image, mask):
        augmentations = self.transform(image=image, mask=mask)
        return augmentations["image"], augmentations["mask"]

    def _apply_preprocessing(self, image, mask):
        sample = self.preprocessing(image=image, mask=mask)
        return sample['image'], sample['mask']


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

# train_transform = A.Compose(
#     [
#         A.Resize(224, 224),
#         A.Normalize(
#             mean=[0.0, 0.0, 0.0],
#             std=[1.0, 1.0, 1.0],
#             max_pixel_value=255.0,
#         ),
#         ToTensorV2(),
#     ],
#     # is_check_shapes=False
# )

train_transform = A.Compose(
    [
        A.Resize(224, 224),
        # A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
        # A.VerticalFlip(p=0.5),    # Randomly flip images vertically
        # A.RandomRotate90(p=0.5),  # Randomly rotate images by 90 degrees
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        # A.GridDistortion(p=0.5),
        # A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(p=1.0, contrast_limit=0.3),
        # A.HueSaturationValue(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]
)

valid_transform = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            # max_pixel_value=255.0,
        ),
        # ZNormalization(),
        ToTensorV2(),
    ],
)

# Train_data = MultiSequenceMRIDataset(Config.all_dir, transform=valid_transform, return_original=True)
# Train_loader = DataLoader(Train_data, batch_size=16, shuffle=True)
# for loader in Train_loader:
#     print(loader[0].shape)
#     break