from sklearn.model_selection import GroupKFold
import numpy as np

# Example simplified dataset
images = ['img1', 'img2', 'img3', 'img4', 'img5', 'img6']
labels = [0, 0, 1, 1, 2, 2]
sample_ids = ['A', 'A', 'B', 'B', 'C', 'C']

group_kfold = GroupKFold(n_splits=3)
group_kfold_iter = group_kfold.split(images, labels, sample_ids)

for fold, (train_idx, test_idx) in enumerate(group_kfold_iter, 1):
    print(f"Fold {fold}:")
    print(f"Train indices: {train_idx}")
    print(f"Test indices: {test_idx}")
    print(f"Train sample IDs: {[sample_ids[i] for i in train_idx]}")
    print(f"Test sample IDs: {[sample_ids[i] for i in test_idx]}\n")