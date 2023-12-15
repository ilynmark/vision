# Image Segmentation Project

This project focuses on image segmentation using a modified U-Net model on the Oxford-IIIT Pet Dataset.

## Overview

In an image segmentation task, the goal is to assign a class to each pixel in an image. This project uses a U-Net architecture with a MobileNetV2 encoder for segmenting pet images into three classes: pixels belonging to the pet, pixels bordering the pet, and surrounding pixels.

## Getting Started
### Prerequisites

- Python 3.x
- TensorFlow
- TensorFlow Datasets
- Matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

# Dataset
The Oxford-IIIT Pet Dataset is used for training and testing. The dataset is automatically downloaded using TensorFlow Datasets, eliminating the need for manual download and folder management.

```python
# The dataset is loaded directly from TensorFlow Datasets
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
```

# Project Structure
- Code Folder:

-- dataset.py: Handles dataset-related tasks, including preprocessing, augmentation, and post-processing.
-- model.py: Defines the U-Net model architecture, layers, and loss functions.
-- train.py: Contains training code. Accepts dataset paths and hyperparameters as inputs, producing and saving checkpoints.
-- inference.py: Contains model inference code for processing a single image, saving the output in the "Result" folder.
- Data Folder:

-- training: Contains data used for training.
-- validation: Contains data used for validation.
-- test: Contains data used for testing.
- Result Folder:
-- Contains the results of the testing phase.


# Usage
## 1. Training:
You can use train.py for training the model based on existing masks dataset.
## 2. Inference:
You can use inference.py to process any image and save output in the result folder.

# Visualization
- Utilize the provided Jupyter notebook or script to visualize training and validation loss.

# Results
- View the "Result" folder for segmentation results on test images.

# Acknowledgments
- This project is based on the TensorFlow Examples repository and the Oxford-IIIT Pet Dataset.