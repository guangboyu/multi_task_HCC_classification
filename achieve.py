class MRIDataset_Unet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = self.load_images_and_labels()

    def load_images_and_labels(self):
        images = []
        labels = []
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
                    for img_path in glob.glob(
                            os.path.join(class_path, patient_id, "*Separate*.nii.gz")
                    ):
                        mask_path = img_path.replace("Separate", "MASK")
                        mask_number = extract_mask_number(mask_path)
                        mask_path = os.path.join(
                            class_path, patient_id, "MASK_" + mask_number + ".nii.gz"
                        )
                        if os.path.isfile(img_path) and os.path.isfile(mask_path):
                            images.append((img_path, mask_path))
                            labels.append(label_int)
                # Config.num_classes = len(label_dir)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, mask_path = self.images[idx]
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)

        # image = torch.from_numpy(image)

        # if len(image.shape) == 2:  # for 2D images
        #     image = image.unsqueeze(0).repeat(3, 1, 1)

        # First, add an extra dimension at the beginning (axis=0)
        image = np.expand_dims(image, axis=2)

        # Then, repeat the array 3 times along the first dimension (axis=0)
        image = np.repeat(image, 3, axis=2)

        if image.shape[:2] != mask.shape:
            print(f"{img_path} has image shape: {image.shape}")
            print(f"{mask_path} has image shape: {mask.shape}")
        mask[mask > 0.0] = 1.0
        # print(set(mask.flatten()))

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

    def binarization(self, label_dir):
        if label_dir in ["Combination", "Sorafenib", "NK"]:
            return "Treatment"
        else:
            return "Control"