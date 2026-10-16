# MNIST Handwritten Digit Classification with MLP

This repository contains an end-to-end machine learning workflow to classify handwritten digits (0–9) using the MNIST dataset and a Multi-Layer Perceptron (MLP) neural network implemented with Scikit-Learn.

## Project Highlights
* **Dataset**: MNIST 784 via OpenML (70,000 grayscale digit images of 28x28 pixels).
* **Data Processing**: Pixel intensity normalization to [0.0, 1.0] and stratified train/test splitting (60k train / 10k test).
* **Architecture**: Deep Neural Network (`MLPClassifier`) with two hidden layers (128 and 64 neurons), ReLU activation, Adam optimizer, and a batch size of 256.
* **Regularization**: Integrated L2 penalty (alpha = 0.001) and automated early stopping using validation scoring to prevent overfitting.
* **Evaluation**: Assessed using accuracy, precision, recall, F1-scores, and a Confusion Matrix. Includes visual inspection of both correct predictions and edge-case misclassifications.

## Results
* **Training Accuracy**: 99.55%
* **Testing Accuracy**: 97.84%
* **Overall F1-Score**: 0.98 across all 10 digit classes

## Tech Stack
* Python
* Scikit-Learn
* NumPy
* Matplotlib
