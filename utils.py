import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, mutual_info_score, accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
import re
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import glob
import json
import matplotlib

import nibabel as nib
import torch
import torchvision
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders(dataset, train_ids, val_ids, augment=False):
    train_subs = Subset(dataset, train_ids)
    val_subs = Subset(dataset, val_ids)
    train_loader = DataLoader(train_subs, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subs, batch_size=Config.batch_size, shuffle=False)
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    """
    Old evaluation
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def get_last_part(feature_name):
    return feature_name.split("_")[1:]


def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    """Compute bootstrap estimates for AUC and its 95% CI."""
    rng = np.random.default_rng(rng_seed)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = rng.choice(range(len(y_pred)), replace=True, size=len(y_pred))

        # Continue bootstrapping if we don't have both classes
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    # Computing the lower and upper bound of the 95% confidence interval
    sorted_scores = np.array(bootstrapped_scores)
    confidence_lower = np.percentile(sorted_scores, 2.5)
    confidence_upper = np.percentile(sorted_scores, 97.5)

    return np.mean(bootstrapped_scores), confidence_lower, confidence_upper


def sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity and specificity."""
    # Get confusion matrix
    print(y_true)
    print(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def extract_mask_number(mask_path):
    # Pattern to find 'MASK_' followed by one or more digits
    pattern = re.compile(r"MASK_(\d+)")
    match = pattern.search(mask_path)
    if match:
        # Extract the first group of digits after 'MASK_'
        return match.group(1)
    else:
        return None


def iou_score(output, target):
    smooth = 1e-6
    if torch.is_tensor(output):
        output = torch.sigmoid(output) > 0.5
        target = target > 0.5
        intersection = (output & target).float().sum((1, 2))
        union = (output | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def dice_score(output, target, smooth=1e-6):
    if torch.is_tensor(output):
        output = torch.sigmoid(output) > 0.5
        target = target > 0.5
        intersection = (output & target).float().sum()
        dice = (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
    return dice


def dice_score_batch(output, target, smooth=1e-6):
    # Assuming the output shape is [batch, channels, height, width]
    if torch.is_tensor(output):
        output = torch.sigmoid(output) > 0.5  # Convert logits to binary
        target = target > 0.5  # Ensure target is binary as well

        intersection = (output & target).float().sum((2, 3))  # Sum over height and width
        dice = (2 * intersection + smooth) / (output.float().sum((2, 3)) + target.float().sum((2, 3)) + smooth)
        return dice.mean()  # Average over the batch


# Function to compute mutual information score
def calculate_mutual_info(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    return mutual_info_score(pred_flat, target_flat)


def compute_confidence_intervals(true_labels, predictions, num_classes, num_bootstraps=1000, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = {'auroc_ci': [], 'sensitivity_ci': [], 'specificity_ci': [], 'accuracy_ci': []}

    for i in range(num_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(predictions), len(predictions))
        if len(np.unique(true_labels[indices])) < 2:
            # We need at least one example of each class to compute ROC AUC
            continue

        y_true = true_labels[indices]
        y_pred = predictions[indices]

        # Ensure y_true is in multiclass format
        y_true_multiclass = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

        # Compute AUROC
        try:
            auroc = roc_auc_score(y_true, y_pred, multi_class='ovr')
            bootstrapped_scores['auroc_ci'].append(auroc)
        except ValueError:
            continue

        # Compute accuracy
        accuracy = accuracy_score(y_true_multiclass, np.argmax(y_pred, axis=1))
        bootstrapped_scores['accuracy_ci'].append(accuracy)

        # Compute sensitivity and specificity
        cm = confusion_matrix(y_true_multiclass, np.argmax(y_pred, axis=1), labels=np.arange(num_classes))
        sensitivity = np.diag(cm) / np.maximum(1, cm.sum(axis=1))
        specificity = np.diag(cm) / np.maximum(1, cm.sum(axis=0))
        bootstrapped_scores['sensitivity_ci'].append(np.mean(sensitivity))
        bootstrapped_scores['specificity_ci'].append(np.mean(specificity))

    # Compute 95% CI
    confidence_intervals = {}
    for metric in bootstrapped_scores:
        scores = np.array(bootstrapped_scores[metric])
        confidence_intervals[metric] = (np.percentile(scores, 2.5), np.percentile(scores, 97.5))

    return confidence_intervals


def check_metrics(
        loader, model, device, return_images_for_visualization=False, save_path=None, metrics_list=None
):
    model.eval()

    with torch.no_grad():
        total_iou, total_dice, total_accuracy, total_auroc = 0, 0, 0, 0
        n_samples = 0
        all_labels, all_preds = [], []
        total_confusion = np.zeros((Config.num_classes, Config.num_classes), dtype=int)

        for batch_index, (data, targets, labels, original_size, img_paths) in enumerate(loader):
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            labels = labels.to(device)
            original_size = original_size[0]

            # predict
            preds_masks, preds_labels = model(data)

            # save segmentation image result
            if return_images_for_visualization:
                file_path = os.path.join(
                    save_path, f"Segmentation_Result_{batch_index + 1}.png"
                )
                # save_segmentation_visualization(
                #     data.cpu()[0], preds_masks.cpu()[0], targets.cpu()[0], file_path
                # )

            # Calculate metrics
            iou = iou_score(preds_masks, targets).cpu().item()
            dice = dice_score(preds_masks, targets).cpu().item()
            total_iou += iou
            total_dice += dice

            # Save metrics per sample
            for i in range(len(img_paths)):
                # Calculate metrics for each sample
                iou_temp = iou_score(preds_masks[i], targets[i]).cpu().item()
                dice_temp = dice_score(preds_masks[i], targets[i]).cpu().item()
                # mutual_info = calculate_mutual_info(preds_masks[i].cpu().numpy(), targets[i].cpu().numpy())

                # Save metrics per sample
                if 'image_path' in metrics_list and iou_temp < metrics_list['iou']:
                    continue
                # Extract type from path
                type_name = os.path.basename(os.path.dirname(os.path.dirname(img_paths[i])))
                metrics_list.append({'image_path': img_paths[i], 'iou': iou_temp, 'dice': dice_temp,
                                     'type': type_name})

            # accuracy
            # Convert logits to class indices for accuracy
            preds_labels_indices = torch.argmax(preds_labels, dim=1).cpu()
            accuracy = accuracy_score(
                labels.cpu().numpy(), preds_labels_indices.numpy()
            )
            total_accuracy += accuracy

            # For AUROC, collect all predictions and labels
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.softmax(preds_labels, dim=1).cpu().numpy())

            # Update the confusion matrix
            conf_matrix = confusion_matrix(
                labels.cpu().numpy(),
                preds_labels_indices.numpy(),
                labels=np.arange(Config.num_classes),
            )
            total_confusion += conf_matrix

            n_samples += 1

        avg_iou = total_iou / n_samples
        avg_dice = total_dice / n_samples
        avg_accuracy = total_accuracy / n_samples

        # Flatten all collected labels and predictions for AUROC calculation
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        # Binarize labels in a one-vs-all manner for AUROC
        all_labels_bin = label_binarize(all_labels, classes=range(Config.num_classes))

        if Config.num_classes == 2:  # binary case, need to adjust shape
            all_labels_bin = all_labels_bin[:, 1]  # take the second column for positive class
        avg_auroc = roc_auc_score(all_labels_bin, all_preds, multi_class="ovr")

        sensitivity = np.diag(total_confusion) / np.maximum(
            1, total_confusion.sum(axis=1)
        )
        specificity = np.diag(total_confusion) / np.maximum(
            1, total_confusion.sum(axis=0)
        )

        print("Average iou: ", avg_iou)
        print("Average dice: ", avg_dice)
        print("Average accuracy: ", avg_accuracy)
        print("Average auroc: ", avg_auroc)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

        metrics = {
            "avg_iou": avg_iou,
            "avg_dice": avg_dice,
            "avg_accuracy": avg_accuracy,
            "avg_auroc": avg_auroc,
            "true_labels_bin": all_labels_bin,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "avg_sensitivity": np.mean(sensitivity),
            "avg_specificity": np.mean(specificity),
            "all_preds": all_preds,
            "total_confusion": total_confusion
        }

        confidence_intervals = compute_confidence_intervals(all_labels_bin, all_preds, Config.num_classes)
        metrics.update(confidence_intervals)
    model.train()
    return metrics


def plot_multiclass_roc(true_labels_bin, all_preds, num_classes, save_path=None, label_names=None):
    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        true_labels_bin.ravel(), all_preds.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= num_classes

    mean_fpr = all_fpr
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot all ROC curves
    plt.figure()

    # plt.plot(mean_fpr, mean_tpr, label='Mean ROC (area = {0:0.2f})'.format(mean_auc), color='navy', linestyle=':',
    #          linewidth=4)
    plt.plot(
        mean_fpr,
        mean_tpr,
        label="Mean: {0:0.2f}".format(mean_auc),
        linewidth=2,
    )

    for i in range(num_classes):
        # plt.plot(fpr[i], tpr[i], lw=2, label='class {0} AUC : {1:0.2f}'.format(i, roc_auc[i]), linestyle="dotted")
        label = label_names[i] if label_names else f'class {i}'
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{label}: {roc_auc[i]:0.2f}', linestyle="dotted")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.figure(figsize=(10, 8))
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("Deep Learning Multi-class ROC")
    plt.legend(loc="lower right", fontsize="small")
    plt.savefig("Figures/" + save_path, dpi=600, bbox_inches="tight")
    plt.show()

    return mean_auc


def save_segmentation_visualization(image, pred_mask, target_mask, file_path):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Assuming the input image is in channels-first format, convert it to channels-last
    if image.shape[0] == 3:  # This checks if there are three channels
        image = image.permute(
            1, 2, 0
        )  # Rearrange dimensions to (height, width, channels)
    else:
        image = image.squeeze()  # Remove channel dimension for grayscale

    probs_masks = torch.sigmoid(pred_mask)  # Convert logits to probabilities
    binary_masks = (probs_masks > 0.5).float()  # Apply threshold to get binary mask

    pred_mask = binary_masks.squeeze().cpu()
    target_mask = target_mask.squeeze().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")  # Adjust .squeeze() as needed
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")  # Squeeze if necessary
    axes[1].imshow(pred_mask, cmap="jet", alpha=0.5)  # Overlay predicted mask
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(target_mask, cmap="jet", alpha=0.5)  # Overlay ground truth mask
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.savefig(file_path)
    plt.close(fig)


def save_predicted_masks(loader, model, device, fold):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _, _, original_size, img_paths) in enumerate(loader):
            data = data.to(device=device)
            pred_mask, _ = model(data)
            pred_mask = torch.sigmoid(pred_mask).cpu().numpy()

            for i, path in enumerate(img_paths):
                save_path = path.replace("Separate", f"Separate_pred_fold{fold}")
                pred_mask_resized = cv2.resize(pred_mask[i, 0], (
                int(original_size[i]), int(original_size[i])))  # Resize to original size
                pred_mask_img = (pred_mask_resized > 0.5).astype(np.uint8)
                pred_mask_nib = nib.Nifti1Image(pred_mask_img, affine=None)
                nib.save(pred_mask_nib, save_path)
    model.train()


def remove_pred_mask():
    # Define the pattern for the files to be deleted
    # pattern = 'Data/Buffalo_SD/*/*/*pred_fold*.nii.gz'
    pattern = Config.all_dir + "/*/*/*pred_fold*.nii.gz"

    # Find all files matching the pattern
    files_to_delete = glob.glob(pattern)

    # Loop through the list of files and delete each one
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
