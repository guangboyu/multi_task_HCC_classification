import glob


class Config:
    data_paths = glob.glob("Data/*/*/*/")
    buffalo_dir = "Data/Buffalo_Remove_Wk1_2024_0311"
    sd_dir = "Data/SD_remove_wk1"
    all_dir = "Data/BF_SD_all_slices_0610"
    num_classes = 4
    num_epochs = 30
    batch_size = 16

    Segmentation_Architecture = [
                                 # "UNet++",
                                 # "MAnet",
                                 "DeepLabV3Plus",
                                 # "FPN",
                                 # "PSPNet",
                                 # "Linknet",
                                 # "PAN"
                                ]
    ENCODER = 'efficientnet-b0'
    ENCODER_WEIGHTS = 'imagenet'

    all_train = "Data/All_Slice_Train"
    all_test = "Data/All_Slice_Test"

    # segmentation loss weight
    alpha = 0.5
    # segmentation loss mode
    segmentation_loss = "binary"