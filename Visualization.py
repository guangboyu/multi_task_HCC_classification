import glob

import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import os
import nibabel as nib
import numpy as np
import pandas as pd

from data_process import extract_mask_number

def visualize_mask_overlap():
    def load_nifti_image(file_path):
        return nib.load(file_path).get_fdata()
    def rotate_nifti_image(image, angle):
        return rotate(image, angle, axes=(0, 1), reshape=False)

    data_path = 'Figures/Segmentation_Examples_Manual/B33R4_091922'
    t2w_path = os.path.join(data_path, 'T2W_HR_Separate_2.nii.gz')
    ground_truth_path = os.path.join(data_path, 'MASK_2.nii.gz')
    predicted_path = os.path.join(data_path, 'T2W_HR_Separate_pred_fold0_2.nii.gz')
    output_path = os.path.join(data_path, 'overlap_visualization.png')

    t2w_img = load_nifti_image(t2w_path)
    ground_truth_mask = load_nifti_image(ground_truth_path)
    predicted_mask = load_nifti_image(predicted_path)

    # Ensure all images are of the same shape
    assert t2w_img.shape == ground_truth_mask.shape == predicted_mask.shape, "Shape mismatch between images and masks"

    t2w_img = t2w_img.astype(np.float32)
    ground_truth_mask = ground_truth_mask.astype(np.float32)
    predicted_mask = predicted_mask.astype(np.float32)

    # # Rotate images and masks
    # angle = 90  # Define the rotation angle
    # t2w_img = rotate_nifti_image(t2w_img, angle)
    # ground_truth_mask = rotate_nifti_image(ground_truth_mask, angle)
    # predicted_mask = rotate_nifti_image(predicted_mask, angle)

    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Display the T2W image
    ax.imshow(t2w_img, cmap='gray')

    # # Create a shadow effect by overlaying the masks with alpha
    # ax.imshow(np.ma.masked_where(ground_truth_mask == 0, ground_truth_mask), cmap='Reds', alpha=0.3)
    # ax.imshow(np.ma.masked_where(predicted_mask == 0, predicted_mask), cmap='Blues', alpha=0.3)

    # Overlay the ground truth mask in red
    ax.contour(ground_truth_mask, colors='r', alpha=0.5, linewidths=0.5)

    # Overlay the predicted mask in blue
    ax.contour(predicted_mask, colors='b', alpha=0.5, linewidths=0.5)

    # Add a legend
    handles = [plt.Line2D([0], [0], color='r', lw=2, label='Ground Truth'),
               plt.Line2D([0], [0], color='b', lw=2, label='Predicted')]
    ax.legend(handles=handles, loc='upper right')

    # Remove x and y-axis numbers
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    # ax.set_title('Overlap between Ground Truth and Predicted Mask')

    plt.savefig(output_path, dpi=600)

    # Show the plot
    plt.show()


def visualize_iou_dice():
    output_path = "Figures/iou_dice.png"
    # Load all sheets from the Excel file
    file_path = "Figures/Record/segmentation_metrics.xlsx"
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names

    # Initialize a dictionary to hold data from each sheet
    data_dict = {}

    # Read each sheet into the dictionary
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        data_dict[sheet] = remove_outliers(df)

    # Define colors for each group
    colors = {
        'Combination': 'red',
        'Control': 'blue',
        'NK': 'green',
        'Sorafenib': 'purple'
    }

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    for sheet_name, df in data_dict.items():
        plt.scatter(df['iou'], df['dice'], alpha=0.5, label=sheet_name, color=colors[sheet_name])

    # Calculate the minimum value for iou and dice
    min_iou = min(min(df['iou']) for df in data_dict.values())
    min_dice = min(min(df['dice']) for df in data_dict.values())
    min_val = min(min_iou, min_dice)

    # Add a 45-degree reference line
    lims = [min_val, 1]  # limits for both axes
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    plt.xlabel('Intersection over Union (IoU)')
    plt.ylabel('Dice Score')
    plt.title('IOU vs DICE for each sample by group')
    plt.legend(title='Group')
    plt.grid(True)

    plt.savefig(output_path, dpi=600)

    plt.show()


# Function to remove outliers using IQR
def remove_outliers(df):
    Q1 = df[['iou', 'dice']].quantile(0.25)
    Q3 = df[['iou', 'dice']].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[~((df[['iou', 'dice']] < (Q1 - 1.5 * IQR)) | (df[['iou', 'dice']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return filtered_df

def visualize_mask():
    """
    show representative mri image
    """
    def load_nifti_image(file_path):
        return nib.load(file_path).get_fdata()
    def visualize_seqs(data_path):
        visualize_single(data_path, pattern="T1")
        visualize_single(data_path, pattern="T2")
        visualize_single(data_path, pattern="T1_PC")

    def visualize_single(data_path, pattern):
        if pattern == "T1":
            path_pattern = "T1W_HR_Separate*.nii.gz"
        elif pattern == "T2":
            path_pattern = "T2W_HR_Separate*.nii.gz"
        elif pattern == "T1_PC":
            path_pattern = "T1W_PC_Separate*.nii.gz"
        else:
            return

        mri_paths = os.path.join(data_path, path_pattern)
        if glob.glob(mri_paths):
            mri_path = glob.glob(mri_paths)[0]
        else:
            print(f"error on {data_path}, {pattern}")
            print(mri_paths)
            print("----------------------------------------")
            return

        Mask_path = glob.glob(os.path.join(data_path, "MASK_*.nii.gz"))[0]
        output_path = os.path.join(data_path, pattern + '.png')
        mri_img = nib.load(mri_path).get_fdata().astype(np.float32)
        ground_truth_mask = nib.load(Mask_path).get_fdata().astype(np.float32)

        # Create a figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Display the T2W image
        ax.imshow(mri_img, cmap='gray')
        # Overlay the ground truth mask in red
        ax.contour(ground_truth_mask, colors='r', alpha=0.5, linewidths=0.5, corner_mask=False)
        # Remove x and y-axis numbers
        ax.axis('off')
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', pad_inches=0)
        plt.show()


    data_path = 'Figures/MRI_Representative_Images/*'
    files = glob.glob(data_path)
    for file in files:
        if "Results" in file:
            continue
        visualize_seqs(file)




if __name__ == '__main__':
    font = {"size": 18}
    matplotlib.rc("font", **font)
    # visualize_mask_overlap()
    # visualize_iou_dice()
    visualize_mask()