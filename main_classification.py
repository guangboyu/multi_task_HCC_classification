import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms
import os
import timm
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import pandas as pd
from datetime import datetime

# from monai.networks.nets import UNet
import segmentation_models_pytorch as smp

from config import Config
from data_process import (
    MRIDataset_Unet_Classification,
    train_transform,
    val_transforms,
)

# from model import UNET
from utils import (
    extract_mask_number,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    check_metrics,
    plot_multiclass_roc,
    save_predicted_masks,
    remove_pred_mask
)

CLASSIFICATION = True

aux_params = dict(
    pooling="avg",  # one of 'avg', 'max'
    dropout=0.5,  # dropout ratio, default is None
    activation="sigmoid",  # activation function, default is None
    classes=Config.num_classes,  # define number of output labels
)


def train_cross_validation():
    dataset = MRIDataset_Unet_Classification(
            Config.all_dir, transform=train_transform, return_original=True
        )

    print("length of dataset", len(dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print("Model is running on:", device)

    # kfold = KFold(n_splits=5, shuffle=True)
    group_kfold = GroupKFold(n_splits=5)

    fold_perf = []
    all_labels = []
    all_preds = []

    # List to store metrics for each sample
    metrics_list = []

    # for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    for fold, (train_ids, val_ids) in enumerate(group_kfold.split(dataset.images, dataset.labels, dataset.sample_ids)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        # Create train dataloader
        train_subs = Subset(dataset, train_ids)
        train_loader = DataLoader(train_subs, batch_size=16, shuffle=True)

        val_subs = Subset(dataset, val_ids)
        val_loader = DataLoader(val_subs, batch_size=16, shuffle=False)

        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
        ## multi-class segmentation
        # segmentation_loss_fn = torch.nn.CrossEntropyLoss()

        # binary mask segmentation
        segmentation_loss_fn = torch.nn.BCEWithLogitsLoss()

        # dice loss segmentation
        # segmentation_loss_fn = smp.losses.DiceLoss(mode="binary")

        classification_loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(
            Config.num_epochs
        ):  # Include epochs in the training function call
            train_fn(
                train_loader,
                model,
                optimizer,
                segmentation_loss_fn,
                scaler,
                device,
                classification_loss_fn=classification_loss_fn,
            )
        visualization_path = f"Figures/Segmentation_Results/Fold_{fold+1}"
        metrics = check_metrics(
            val_loader,
            model,
            device=device,
            return_images_for_visualization=True,
            save_path=visualization_path,
            metrics_list=metrics_list
        )
        fold_perf.append(metrics)
        all_labels.append(metrics["true_labels_bin"])
        all_preds.append(metrics["all_preds"])

        save_predicted_masks(val_loader, model, device, fold)

    # Plotting the average ROC curve from folds
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    plot_multiclass_roc(
        all_labels,
        all_preds,
        Config.num_classes,
        save_path="deep_learning_multi_class_roc.png",
    )

    # Print average metrics
    # Compute average metrics correctly
    avg_iou = np.mean([f["avg_iou"] for f in fold_perf])
    avg_dice = np.mean([f["avg_dice"] for f in fold_perf])
    avg_accuracy = np.mean([f["avg_accuracy"] for f in fold_perf])
    avg_auroc = np.mean([f["avg_auroc"] for f in fold_perf])
    avg_sensitivity = np.mean([f["avg_sensitivity"] for f in fold_perf])
    avg_specificity = np.mean([f["avg_specificity"] for f in fold_perf])
    print("5-Fold Average IOU:", avg_iou)
    print("5-Fold Average Dice:", avg_dice)
    print("5-Fold Average Accuracy:", avg_accuracy)
    print("5-Fold Average AUROC:", avg_auroc)
    print("Sensitivity: ", )
    print("Average Sensitivity:", avg_sensitivity)
    print("Average Specificity:", avg_specificity)

    # Save average metrics to a text file
    with open('Figures/average_metrics.txt', 'w') as f:
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"5-Fold Average IOU: {avg_iou}\n")
        f.write(f"5-Fold Average Dice: {avg_dice}\n")
        f.write(f"5-Fold Average Accuracy: {avg_accuracy}\n")
        f.write(f"5-Fold Average AUROC: {avg_auroc}\n")
        f.write(f"Average Sensitivity: {avg_sensitivity}\n")
        f.write(f"Average Specificity: {avg_specificity}\n")

    # Save metrics to Excel in different sheets based on type
    metrics_df = pd.DataFrame(metrics_list)
    with pd.ExcelWriter('Figures/segmentation_metrics.xlsx') as writer:
        for type_name in metrics_df['type'].unique():
            metrics_df[metrics_df['type'] == type_name].to_excel(writer, sheet_name=type_name, index=False)


def train_fn(
    loader, model, optimizer, loss_fn, scaler, device, classification_loss_fn=None
):
    loop = tqdm(loader)

    for batch_idx, (data, targets, labels, original_size, img_paths) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        labels = labels.to(device=device)

        with torch.cuda.amp.autocast():
            # predict
            pred_mask, pred_label = model(data)

            seg_loss = loss_fn(pred_mask, targets)
            label_loss = classification_loss_fn(pred_label, labels)
            total_loss = Config.alpha * seg_loss + (1 - Config.alpha) * label_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=total_loss.item())



if __name__ == "__main__":
    font = {"size": 20}
    matplotlib.rc("font", **font)
    torch.manual_seed(42)
    remove_pred_mask()
    train_cross_validation()
