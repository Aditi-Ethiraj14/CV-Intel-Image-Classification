Intel Image Classification - Computer Vision Project
Overview

This project demonstrates an Image Classification task using deep learning with PyTorch. The goal is to classify images from the Intel Image Classification dataset into six categories:

buildings

forest

glacier

mountain

sea

street

The project covers the full workflow: dataset understanding → preprocessing → model building → evaluation → visualization of results.

Dataset

The dataset used in this project is the Intel Image Classification dataset from Kaggle:
Kaggle Dataset Link

Structure
intel_data/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_test/
    ├── buildings/
    ├── forest/
    ├── glacier/
    ├── mountain/
    ├── sea/
    └── street/

Project Workflow
1. Data Preprocessing

Resized images to 150×150 pixels.

Converted images to tensors and normalized pixel values.

Created DataLoaders for training and testing datasets.

2. Model Building

Implemented a Convolutional Neural Network (CNN) using PyTorch.

Architecture:

2 convolutional layers + ReLU + MaxPooling

Fully connected layers

Output layer with 6 classes

Used CrossEntropyLoss and Adam optimizer for training.

3. Training

Trained for 10 epochs on the training dataset.

Monitored validation accuracy on the test dataset during training.

Final model saved as intel_model.pth.

4. Evaluation & Visualization

Computed final test accuracy.

Visualized predictions vs true labels for random images from each class.

Displayed images with Predicted (P) and True (T) labels.

Installation / Running the Code

Clone the repository:

git clone <your-repo-url>
cd <repo-folder>/src


Install dependencies:

pip install torch torchvision matplotlib


Data Preprocessing
Run data_preprocessing.py to prepare the dataset and DataLoaders.

Training the Model
Run train_model.py to train the CNN and save the model.

Visualizing Results
Run test_visualize.py to see predictions for one image per class.

Folder Structure
src/
├── data_preprocessing.py   # Dataset preparation and DataLoaders
├── train_model.py          # CNN model training and evaluation
├── test_visualize.py       # Load model & visualize predictions
intel_data/                 # Kaggle dataset folder (after download)
README.md

Results

Achieved X% accuracy on the test set (replace X with your final accuracy).

Sample predictions:

Image	Predicted	True

	buildings	buildings

	forest	forest

	glacier	glacier

(Include screenshots from your test_visualize.py output.)

Skills Demonstrated

Python programming for deep learning.

Using PyTorch for CNN-based image classification.

Understanding and handling image datasets.

Model training, evaluation, and visualization.

GitHub workflow: code organization, README documentation.

Optional Enhancements

Increase model depth for higher accuracy.

Apply data augmentation (flipping, rotation, etc.)

Implement early stopping or learning rate scheduler.

Experiment with pretrained models (ResNet, VGG) for transfer learning.

References

Intel Image Classification Dataset: Kaggle

PyTorch Documentation: https://pytorch.org/docs/stable/index.html
