import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import os
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize


from utils import (
    check_accuracy,
    check_metrics,
    plot_multiclass_roc,
    save_predicted_masks,
    remove_pred_mask,
    dice_score,
    iou_score
)

from config import Config


def evaluate_model(val_loader, model, device, fold, metrics_list):
    visualization_path = f"Figures/Segmentation_Results/Fold_{fold + 1}"
    metrics = check_metrics(
        val_loader,
        model,
        device=device,
        return_images_for_visualization=True,
        save_path=visualization_path,
        metrics_list=metrics_list
    )
    save_predicted_masks(val_loader, model, device, fold)
    return metrics


def save_metrics_to_files(data_path, metrics_list, fold_perf, label_names, total_confusion, model_name="", encoder=Config.ENCODER):
    avg_metrics = {
        'avg_iou': np.mean([f["avg_iou"] for f in fold_perf]),
        'avg_dice': np.mean([f["avg_dice"] for f in fold_perf]),
        'avg_accuracy': np.mean([f["avg_accuracy"] for f in fold_perf]),
        'avg_auroc': np.mean([f["avg_auroc"] for f in fold_perf]),
        'avg_sensitivity': np.mean([f["avg_sensitivity"] for f in fold_perf]),
        'avg_specificity': np.mean([f["avg_specificity"] for f in fold_perf])
    }

    ci_metrics = {
        'auroc_ci': np.mean([f['auroc_ci'] for f in fold_perf], axis=0),
        'accuracy_ci': np.mean([f['accuracy_ci'] for f in fold_perf], axis=0),
        'sensitivity_ci': np.mean([f['sensitivity_ci'] for f in fold_perf], axis=0),
        'specificity_ci': np.mean([f['specificity_ci'] for f in fold_perf], axis=0)
    }
    file_date = datetime.now().strftime('%Y-%m-%d')
    with open(f'Figures/{file_date}_average_metrics.txt', 'a+') as f:
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Model information: {model_name} {encoder}\n")
        for key, value in avg_metrics.items():
            f.write(f"5-Fold Average {key.replace('avg_', '').upper()}: {value}\n")
        for key, value in ci_metrics.items():
            f.write(f"{key.replace('_ci', '').upper()} 95% CI: {value}\n")
        f.write("\n\n")

    metrics_df = pd.DataFrame(metrics_list)
    with pd.ExcelWriter('Figures/segmentation_metrics.xlsx') as writer:
        for type_name in metrics_df['type'].unique():
            metrics_df[metrics_df['type'] == type_name].to_excel(writer, sheet_name=type_name, index=False)

    disp = ConfusionMatrixDisplay(confusion_matrix=total_confusion, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('Figures/combined_confusion_matrix.png', dpi=600)
    plt.close()


def compute_confidence_intervals(true_labels, predictions, num_classes, num_bootstraps=1000, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = {'auroc_ci': [], 'sensitivity_ci': [], 'specificity_ci': [], 'accuracy_ci': []}

    for _ in range(num_bootstraps):
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


def plot_calibration_curves(true_labels, pred_probs, num_classes, label_names=None):
    # print("true_labels: ", true_labels)
    # print("pred_probs: ", pred_probs)
    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        prob_pos = pred_probs[:, i]
        true_pos = (true_labels == i).astype(int)

        fraction_of_positives, mean_predicted_value = calibration_curve(true_pos, prob_pos, n_bins=10)
        label = label_names[i] if label_names else f'class {i}'
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=label)

    # Plot reference line
    plt.plot([0, 1], [0, 1], "k--")

    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.savefig('Figures/calibration_curves.png')


def plot_multiclass_roc(true_labels_bin, all_preds, num_classes, save_path=None, label_names=None):
    # Calculate ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), all_preds.ravel())
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
    plt.plot(mean_fpr, mean_tpr, label="Mean: {0:0.2f}".format(mean_auc), linewidth=2)

    for i in range(num_classes):
        label = label_names[i] if label_names else f'class {i}'
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{label}: {roc_auc[i]:0.2f}', linestyle="dotted")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("Deep Learning Multi-class ROC")
    plt.legend(loc="lower right", fontsize="small")
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    return mean_auc


def check_metrics(loader, model, device, return_images_for_visualization=False, save_path=None, metrics_list=None):
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

            # Predict
            preds_masks, preds_labels = model(data)

            # Save segmentation image result
            if return_images_for_visualization and save_path:
                file_path = os.path.join(save_path, f"Segmentation_Result_{batch_index + 1}.png")
                # save_segmentation_visualization(data.cpu()[0], preds_masks.cpu()[0], targets.cpu()[0], file_path)

            # Calculate metrics
            iou = iou_score(preds_masks, targets).cpu().item()
            dice = dice_score(preds_masks, targets).cpu().item()
            total_iou += iou
            total_dice += dice

            # Save metrics per sample
            for i in range(len(img_paths)):
                iou_temp = iou_score(preds_masks[i], targets[i]).cpu().item()
                dice_temp = dice_score(preds_masks[i], targets[i]).cpu().item()

                if 'image_path' in metrics_list and iou_temp < metrics_list['iou']:
                    continue

                type_name = os.path.basename(os.path.dirname(os.path.dirname(img_paths[i])))
                metrics_list.append({'image_path': img_paths[i], 'iou': iou_temp, 'dice': dice_temp, 'type': type_name})

            # Calculate accuracy
            preds_labels_indices = torch.argmax(preds_labels, dim=1).cpu()
            accuracy = accuracy_score(labels.cpu().numpy(), preds_labels_indices.numpy())
            total_accuracy += accuracy

            # For AUROC, collect all predictions and labels
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.softmax(preds_labels, dim=1).cpu().numpy())

            # Update the confusion matrix
            conf_matrix = confusion_matrix(labels.cpu().numpy(), preds_labels_indices.numpy(), labels=np.arange(Config.num_classes))
            total_confusion += conf_matrix

            n_samples += 1

        # Calculate average metrics
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

        sensitivity = np.diag(total_confusion) / np.maximum(1, total_confusion.sum(axis=1))
        specificity = np.diag(total_confusion) / np.maximum(1, total_confusion.sum(axis=0))

        # Print metrics
        print("Average iou: ", avg_iou)
        print("Average dice: ", avg_dice)
        print("Average accuracy: ", avg_accuracy)
        print("Average auroc: ", avg_auroc)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

        # Create metrics dictionary
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
            "total_confusion": total_confusion,
            "all_labels": all_labels
        }

        # Compute confidence intervals and update metrics
        confidence_intervals = compute_confidence_intervals(all_labels_bin, all_preds, Config.num_classes)
        metrics.update(confidence_intervals)

        # plot_calibration_curves(all_labels, all_preds, Config.num_classes)


    model.train()
    return metrics



