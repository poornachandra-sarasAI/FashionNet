# FashionNet

A deep learning neural network for fashion image classification using TensorFlow and the Fashion MNIST dataset.

## Project Overview

FashionNet is a neural network model built to classify fashion items into 10 distinct categories. The project demonstrates the application of neural networks to image classification problems and explores various optimization techniques to improve model performance.

## Dataset

This project uses the Fashion MNIST dataset, which consists of 60,000 28x28 grayscale images in the training set and 10,000 test images. Each image is labeled with one of 10 fashion categories:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Project Structure

The project is organized into four main notebook files that document the journey from data exploration to the final optimized model:

1. **01_FashionNet_EDA.ipynb**: Exploratory Data Analysis of the Fashion MNIST dataset
2. **02_FashionNet_NN_Training.ipynb**: Initial Neural Network model training
3. **03_FashionNet_Optimization_Tuning.ipynb**: Model optimization and hyperparameter tuning
4. **04_FashionNet_FinalTesting_Eval_Reflection.ipynb**: Final model evaluation and performance reflection

## Model Evolution

The model architecture evolved throughout the project:

1. **Initial Model**: A simple neural network with one hidden layer containing just 2 nodes, using sigmoid activation and MSE loss.
2. **Optimized Model**: An improved model with better architecture, activation functions (ReLU), and optimizers (Adam), resulting in significantly improved accuracy.

## Key Findings

- The project demonstrates how a neural network can effectively classify fashion items with high accuracy.
- Model optimization significantly improved performance - the final model achieves much better accuracy compared to the initial implementation.
- The optimized model shows strong performance across all 10 fashion categories.

## Assets

The repository includes several important model files:

- `assets/FashionNet.pkl`: Pickled dataset file
- `assets/FashionNet.keras`: Initial model weights
- `assets/FashionNet_Optimized.keras`: Optimized model weights

## Getting Started

To run this project:

1. Clone this repository
2. Ensure you have the required dependencies: TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
3. Run the notebooks in sequence to understand the full model development process

## Technical Highlights

- Data preprocessing and normalization
- Neural network architecture design and training
- Hyperparameter tuning and optimization
- Model evaluation using confusion matrices and classification metrics
- Dimensionality reduction through neural network layers

## Reflection

The project demonstrates how neural networks can transform high-dimensional image data (784 dimensions for 28x28 images) into a compact feature space (as small as 3 dimensions in the hidden layer) while maintaining the ability to accurately classify images. This showcases the power of neural networks for feature extraction and classification tasks.

## Author

**Mohan Poornachandra**

## License

This project is licensed under the MIT License - see below for details:

MIT License

Copyright (c) 2025 Mohan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

* Fashion MNIST dataset provided by Zalando Research
* TensorFlow team for their excellent deep learning framework
* MATH201 course instructors and classmates for their support and feedback
