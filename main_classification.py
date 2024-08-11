import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchviz import make_dot
import onnx
import os
import timm
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import pandas as pd
from datetime import datetime
import ssl

# from monai.networks.nets import UNet
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from config import Config
from data_process import (
    MRIDataset_Unet_Classification,
    MultiSequenceMRIDataset,
    train_transform,
    valid_transform,
)

# from model import UNET
from utils import (
    extract_mask_number,
    load_checkpoint,
    save_checkpoint,
    get_data_loaders,
    get_device,
    check_accuracy,
    check_metrics,
    plot_multiclass_roc,
    save_predicted_masks,
    remove_pred_mask
)
from model import (
    get_model
)
from evaluation import (
    evaluate_model,
    save_metrics_to_files,
    plot_calibration_curves
)

ssl._create_default_https_context = ssl._create_unverified_context
CLASSIFICATION = True

aux_params = dict(
    pooling="avg",  # one of 'avg', 'max'
    dropout=0.5,  # dropout ratio, default is None
    activation="sigmoid",  # activation function, default is None
    classes=Config.num_classes,  # define number of output labels
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device, classification_loss_fn=None):
    loop = tqdm(loader)
    for batch_idx, (data, targets, labels, original_size, img_paths) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        labels = labels.to(device=device)

        with torch.cuda.amp.autocast():
            pred_mask, pred_label = model(data)
            seg_loss = loss_fn(pred_mask, targets)
            label_loss = classification_loss_fn(pred_label, labels)
            total_loss = Config.alpha * seg_loss + (1 - Config.alpha) * label_loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=total_loss.item())


def train_model(train_loader, model, optimizer, segmentation_loss_fn, scaler, device, classification_loss_fn):
    for epoch in range(Config.num_epochs):
        print('Epoch: {}/{}'.format(epoch + 1, Config.num_epochs))
        train_one_epoch(train_loader, model, optimizer, segmentation_loss_fn, scaler, device, classification_loss_fn)


def train_cross_validation(data_path, model_name="", augment=False):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(Config.ENCODER, Config.ENCODER_WEIGHTS)
    # dataset = MRIDataset_Unet_Classification(Config.all_dir, transform=valid_transform, return_original=True)
    dataset = MultiSequenceMRIDataset(data_path, transform=valid_transform, return_original=True)

    print("length of dataset", len(dataset))
    print("sample unique length", len(set(dataset.sample_ids)))
    print("samples: ", dataset.sample_ids)

    device = get_device()
    print("Model is running on:", device)

    kfold = KFold(n_splits=5, shuffle=True)
    kfold_iter = kfold.split(dataset)

    group_kfold = GroupKFold(n_splits=8)
    group_kfold_iter = group_kfold.split(dataset.images, dataset.labels, dataset.sample_ids)

    logo = LeaveOneGroupOut()
    logo_iter = logo.split(dataset.images, dataset.labels, dataset.sample_ids)

    print("dataset images shape", len(dataset.images))
    print("dataset labels shape", len(dataset.labels))

    fold_perf = []
    all_labels = []
    all_preds = []
    all_labels_bin = []
    total_confusion = np.zeros((Config.num_classes, Config.num_classes), dtype=int)
    label_names = list(dataset.label_to_int.keys()) if dataset.label_to_int else []

    metrics_list = []

    for fold, (train_ids, val_ids) in enumerate(kfold_iter):
        print(f"FOLD {fold}")
        print("--------------------------------")

        if augment:
            # Create train and validation subsets with respective transforms (for augumentation)
            train_subs = Subset(MultiSequenceMRIDataset(data_path, transform=train_transform, return_original=True)
                                , train_ids) + Subset(dataset, train_ids)
            val_subs = Subset(dataset, val_ids)
            print("training subset size: ", len(train_subs))
            print("validation subset size: ", len(val_subs))
            train_loader = DataLoader(train_subs, batch_size=Config.batch_size, shuffle=True)
            val_loader = DataLoader(val_subs, batch_size=Config.batch_size, shuffle=False)
        else:
            train_loader, val_loader = get_data_loaders(dataset, train_ids, val_ids)

        model = get_model(device, model_name)

        segmentation_loss_fn = smp.losses.DiceLoss(mode="binary")
        classification_loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scaler = torch.cuda.amp.GradScaler()

        train_model(train_loader, model, optimizer, segmentation_loss_fn, scaler, device, classification_loss_fn)
        metrics = evaluate_model(val_loader, model, device, fold, metrics_list)

        fold_perf.append(metrics)
        all_labels_bin.append(metrics["true_labels_bin"])
        all_labels.append(metrics["all_labels"])
        all_preds.append(metrics["all_preds"])
        total_confusion += metrics["total_confusion"]

    all_labels_bin = np.concatenate(all_labels_bin)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    plot_multiclass_roc(
        all_labels_bin,
        all_preds,
        Config.num_classes,
        save_path=model_name+"_deep_learning_multi_class_roc.png",
        label_names=label_names
    )
    plot_calibration_curves(all_labels, all_preds, Config.num_classes, label_names=label_names)
    save_metrics_to_files(data_path=data_path, metrics_list=metrics_list, fold_perf=fold_perf, label_names=label_names,
                          total_confusion=total_confusion, model_name=model_name)


def train_validation():
    preprocessing_fn = smp.encoders.get_preprocessing_fn(Config.ENCODER, Config.ENCODER_WEIGHTS)
    train_dataset = MRIDataset_Unet_Classification(Config.all_train, transform=valid_transform,
                                                   return_original=True, preprocessing=preprocessing_fn)
    test_dataset = MRIDataset_Unet_Classification(Config.all_test, transform=valid_transform,
                                                  return_original=True, preprocessing=preprocessing_fn)

    print("length of train dataset", len(train_dataset))
    print("length of test dataset", len(test_dataset))

    device = get_device()
    print("Model is running on:", device)

    fold_perf = []
    all_labels = []
    all_preds = []
    total_confusion = np.zeros((Config.num_classes, Config.num_classes), dtype=int)
    label_names = list(train_dataset.label_to_int.keys()) if train_dataset.label_to_int else []

    metrics_list = []

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model = get_model(device)

    segmentation_loss_fn = smp.losses.DiceLoss(mode="binary")
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    train_model(train_loader, model, optimizer, segmentation_loss_fn, scaler, device, classification_loss_fn)
    metrics = evaluate_model(val_loader, model, device, 0, metrics_list)

    fold_perf.append(metrics)
    all_labels.append(metrics["true_labels_bin"])
    all_preds.append(metrics["all_preds"])
    total_confusion += metrics["total_confusion"]

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    plot_multiclass_roc(
        all_labels,
        all_preds,
        Config.num_classes,
        save_path="deep_learning_multi_class_roc.png",
        label_names=label_names
    )
    plot_calibration_curves(all_labels, all_preds, Config.num_classes, label_names=label_names)

    save_metrics_to_files(metrics_list, fold_perf, label_names, total_confusion)


def visualize_model():
    train_dataset = MRIDataset_Unet_Classification(Config.all_train, transform=valid_transform, return_original=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    device = get_device()
    model = get_model(device)
    # print(model)

    for batch_idx, (data, targets, labels, original_size, img_paths) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        labels = labels.to(device=device)
    #
    #     make_dot(model(data), params=dict(model.named_parameters()))
    #     # pred_mask, pred_label = model(data)
    #     break

        with torch.no_grad():
            torch.onnx.export(
                model,
                data,
                'unet.onnx',
                opset_version=11,
                input_names=['input'],
                output_names=['output']
            )
        break

    onnx_model = onnx.load('unet+.onnx')
    onnx.checker.check_model(onnx_model)
    print('ONNX model success')
    print(onnx.helper.printable_graph(onnx_model.graph))


if __name__ == "__main__":
    font = {"size": 18}
    matplotlib.rc("font", **font)
    torch.manual_seed(42)
    remove_pred_mask()

    for model_name in Config.Segmentation_Architecture:
        print(f"Loading model {model_name}")
        remove_pred_mask()
        train_cross_validation(data_path=Config.all_dir, model_name=model_name, augment=True)


    # train_validation()
    # visualize_model()
