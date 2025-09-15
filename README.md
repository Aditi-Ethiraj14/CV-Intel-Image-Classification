# Intel Image Classification - Computer Vision Project

## Overview

This project demonstrates an **Image Classification** task using deep learning with **PyTorch**. The goal is to classify images from the **Intel Image Classification dataset** into six categories:

- buildings  
- forest  
- glacier  
- mountain  
- sea  
- street  

The project covers the full workflow: **dataset understanding → preprocessing → model building → evaluation → visualization of results**.

---

## Dataset

The dataset used in this project is the **Intel Image Classification dataset** from Kaggle:  
[Kaggle Dataset Link](https://www.kaggle.com/puneet6060/intel-image-classification)

---

## Project Workflow

### 1. Data Preprocessing
- Resized images to 150×150 pixels.  
- Converted images to tensors and normalized pixel values.  
- Created **DataLoaders** for training and testing datasets.

### 2. Model Building
- Implemented a **Convolutional Neural Network (CNN)** using PyTorch.  

**Architecture:**
- 2 convolutional layers + ReLU + MaxPooling  
- Fully connected layers  
- Output layer with 6 classes  

- Used **CrossEntropyLoss** and **Adam optimizer** for training.  

### 3. Training
- Trained for **10 epochs** on the training dataset.  
- Monitored **validation accuracy** on the test dataset during training.  
- Final model saved as `intel_model.pth`.  

### 4. Evaluation & Visualization
- Computed final test accuracy.  
- Visualized **predictions vs true labels** for random images from each class.  
- Displayed images with **Predicted (P)** and **True (T)** labels.  

---

## Installation / Running the Code

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-folder>/src
```
2. Install dependencies:
```bash
pip install torch torchvision matplotlib
```
3. Data Preprocessing
Run data_preprocessing.py to prepare the dataset and DataLoaders.

4. Training the Model
Run train_model.py to train the CNN and save the model.

5. Visualizing Results
Run test_visualize.py to see predictions for one image per class.

src/
├── data_preprocessing.py   # Dataset preparation and DataLoaders
├── train_model.py          # CNN model training and evaluation
├── test_visualize.py       # Load model & visualize predictions
intel_data/                 # Kaggle dataset folder (after download)
README.md


Results

Achieved 76.87% accuracy on the test set.

Sample predictions:

| Image   | Predicted | True      |
| ------- | --------- | --------- |
| Sample1 | buildings | buildings |
| Sample2 | forest    | forest    |
| Sample3 | glacier   | glacier   |


<img width="1009" height="590" alt="image" src="https://github.com/user-attachments/assets/c5683a4e-4732-41fe-a73d-1959c02f06ce" />
<img width="389" height="425" alt="image" src="https://github.com/user-attachments/assets/01165a73-9e38-40da-a48a-1cafb7c37bda" />


### References
- **Intel Image Classification Dataset:** [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)  
- **PyTorch Documentation:** [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

