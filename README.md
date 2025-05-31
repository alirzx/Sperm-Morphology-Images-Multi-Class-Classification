# Sperm-Morphology-Images-Multi-Class-Classification
Sperm Morphology Images Multi-Class Classification using Resnet18 model
![](https://github.com/alirzx/Sperm-Morphology-Images-Multi-Class-Classification/blob/main/plots/train_random_samples.png)



Multi-class Sperm Morphology Classification
This project develops a deep learning-based framework for the multi-class classification of sperm morphology using high-resolution microscopic images from a Sperm Morphology Dataset. By employing state-of-the-art convolutional neural network (CNN) architectures, including ResNet18, ResNet50, ResNeXt50, DenseNet201, and EfficientNet-B0, the system classifies sperm images into three categories: normal, abnormal, and unknown. The goal is to provide an automated, accurate, and efficient tool for assessing sperm quality, supporting fertility diagnostics and reproductive health research.
Table of Contents

Project Overview
Dataset
Methods
Installation
Usage
Directory Structure
Requirements
Acknowledgments

Project Overview
Sperm morphology analysis is critical for evaluating male fertility. Traditional methods rely on manual microscopic examination, which is time-consuming and subjective. This project leverages deep learning to automate sperm morphology classification, achieving high accuracy and robustness. The system uses advanced CNN models pre-trained on ImageNet, fine-tuned on the Sperm Morphology Dataset, and optimized for high-resolution microscopic images.
Dataset
The Sperm Morphology Dataset consists of high-resolution microscopic images of sperm, annotated into three classes:

Normal: Sperm with standard morphology, indicating healthy reproductive potential.
Abnormal: Sperm with morphological defects (e.g., head, tail, or midpiece abnormalities).
Unknown: Sperm with ambiguous or unclassifiable features.

The dataset is organized into training and evaluation sets, with approximately 2,592 images per set, balanced across classes. Images are preprocessed into 224x224 RGB PNGs using data augmentation techniques (e.g., resizing, random cropping, flipping) to enhance model generalization.
Methods
The classification pipeline includes:

Preprocessing:
Images are resized to 256x256 and randomly cropped to 224x224.
Data augmentation: random horizontal/vertical flips, affine transformations, color jitter, Gaussian blur, and sharpness adjustment.


Models:
CNN architectures: ResNet18, ResNet50, ResNeXt50, DenseNet201, EfficientNet-B0.
Pre-trained weights from ImageNet are fine-tuned on the Sperm Morphology Dataset.


Training:
Loss function: Cross-Entropy Loss with optional hybrid loss (e.g., triplet loss for ResNeXt50).
Optimizer: Adam with learning rate 1e-4.
Batch size: 32 (adjustable to 16 for GPUs with 6 GB VRAM, e.g., RTX 4050).
Mixed precision training using torch.cuda.amp for efficiency.


Evaluation:
Metrics: Accuracy, precision, recall, F1-score.
Confusion matrix visualization using seaborn.



Installation

Clone the repository:
git clone https://github.com/your-username/sperm-morphology-classification.git
cd sperm-morphology-classification


Create a virtual environment:
python -m venv pytorch_env
pytorch_env\Scripts\activate  # Windows


Install dependencies:
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install numpy==1.24.4 scikit-learn==1.3.2 pillow==10.4.0 matplotlib seaborn jupyter



Usage

Prepare the dataset:

Place the sperm_images_train and sperm_images_eval directories in the project root.
Ensure each directory has subfolders: normal, abnormal, unknown.


Run the notebook:
jupyter notebook

Open train_sperm_classification.ipynb and execute the cells to:

Load and preprocess data.
Train the selected CNN model.
Evaluate performance and visualize results.


Example command for training:
python train.py --model resnext50 --data-dir sperm_images_train --batch-size 32 --epochs 50


Evaluate a pre-trained model:
python evaluate.py --model resnext50 --checkpoint resnext50_epoch_50.pth --data-dir sperm_images_eval



Directory Structure
sperm-morphology-classification/
├── sperm_images_train/           # Training images (normal/, abnormal/, unknown/)
├── sperm_images_eval/            # Evaluation images (normal/, abnormal/, unknown/)
├── models/                       # Pre-trained model checkpoints
│   ├── resnext50_epoch_50.pth
│   ├── densenet201_epoch_32.pth
│   └── ...
├── notebooks/                    # Jupyter notebooks
│   ├── train_sperm_classification.ipynb
│   ├── evaluate_sperm_classification.ipynb
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── utils.py                      # Utility functions
├── README.md                     # Project documentation
└── requirements.txt              # Dependency list

Requirements

Hardware: GPU recommended (e.g., NVIDIA RTX 4050 with 6 GB VRAM).
Software:
Python 3.9
PyTorch 2.6.0+cu126
torchvision 0.21.0+cu126
numpy 1.24.4
scikit-learn 1.3.2
pillow 10.4.0
matplotlib
seaborn
jupyter


OS: Windows (Linux/macOS compatible with minor adjustments).

See requirements.txt for a complete list:
pip install -r requirements.txt

Acknowledgments

Sperm Morphology Dataset: Hypothetical dataset inspired by medical imaging research.
PyTorch: For providing the deep learning framework.
torchvision: For pre-trained models and data augmentation tools.
scikit-learn: For evaluation metrics and label encoding.

For questions or contributions, please contact alirahshmi@gmail.com.
