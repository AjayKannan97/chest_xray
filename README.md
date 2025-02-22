# Chest X-ray Classification

## Overview

This repository contains the implementation of a deep learning-based approach for classifying chest X-ray images to detect pneumonia. The project leverages convolutional neural networks (CNNs) to achieve high accuracy in distinguishing between normal and pneumonia-infected lungs.

## Dataset 

The dataset used for this project is the Chest X-ray dataset from the Kaggle dataset or NIH dataset, which consists of:

- Normal images (healthy chest X-rays)

- Pneumonia images (X-rays indicating pneumonia)

## Model Architecture

The classification model is built using Convolutional Neural Networks (CNNs). The model includes:

- Convolutional layers for feature extraction

- Batch normalization to stabilize training

- Max pooling layers to reduce spatial dimensions

- Fully connected layers for final classification

- Softmax activation for probability distribution

## Libraries and Frameworks

- Python

- TensorFlow/Keras

- OpenCV

- NumPy

- Matplotlib

## Installation

To set up the project locally, follow these steps:

## Clone the repository:

```
git clone https://github.com/AjayKannan97/chest_xray.git
cd chest_xray
```

## Install the required dependencies:

pip install -r requirements.txt

Download the dataset and place it in the appropriate directory.

## Training the Model

Run the following command to train the model:

``` python train.py ```

This will initiate the training process and save the best model weights.

## Testing the Model

To test the trained model, use:

```
python test.py
```

This will evaluate the model's accuracy on the test dataset.

## Results

Achieved high classification accuracy on test data.

Used data augmentation techniques to improve generalization.

Fine-tuned CNN architectures for optimal performance.

## Future Improvements

Implement additional architectures such as ResNet, EfficientNet.

Explore transfer learning to further improve accuracy.

Deploy the model as a web application for real-time chest X-ray classification.

## Contributors

Ajay Kannan (@AjayKannan97)

## License

This project is licensed under the MIT License.
