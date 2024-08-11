
---

# Multi-Task Deep Learning Framework for Tumor Segmentation and Treatment Outcome Classification

This repository contains the implementation of a multi-task deep learning framework designed for segmenting tumor regions and classifying treatment outcomes in medical imaging. The project uses U-Net++ with a classification head for these tasks, specifically applied to hepatocellular carcinoma (HCC) in a rat model.

## Project Structure

- **config.py**: Configuration file containing parameters and settings for the project.
- **data_process.py**: Scripts for data preprocessing, augmentation, and loading.
- **evaluation.py**: Evaluation metrics and functions to assess the performance of the model.
- **main_classification.py**: Main script for training and testing the classification model.
- **model.py**: Definition of the U-Net++ model architecture with the classification head.
- **utils.py**: Utility functions used throughout the project.

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/guangboyu/multi_task_HCC_classification.git
cd multi_task_HCC_classification
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Prepare your dataset and ensure it is structured correctly. Update the paths and parameters in `config.py` accordingly.

### Training the Model

To train the model, run the `main_classification.py` script:

```bash
python main_classification.py
```

This will start the training process using the configurations specified in `config.py`.


## Configuration

The `config.py` and `model.py` file contains various parameters that can be adjusted to fine-tune the model:

- **Data paths**
- **Model hyperparameters**
- **Training settings**
- **Evaluation settings**

## Model Architecture

The model is based on U-Net++, a powerful architecture for medical image segmentation. A classification head is added to perform the treatment outcome classification.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to the UCI Translational Lab team and contributors who provided invaluable input and resources for this project.
