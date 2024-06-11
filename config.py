import glob


class Config:
    data_paths = glob.glob("Data/*/*/*/")
    buffalo_dir = "Data/Buffalo_Remove_Wk1_2024_0311"
    sd_dir = "Data/SD_remove_wk1"
    all_dir = "Data/BF_SD_all_slices_0610"
    num_classes = 4
    num_epochs = 30

    # segmentation loss weight
    alpha = 0.9
    # segmentation loss mode
    segmentation_loss = "binary"