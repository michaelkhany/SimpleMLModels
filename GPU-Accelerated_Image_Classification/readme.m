# GPU-Accelerated Image Classification with CNN on CIFAR-10 Dataset

This repository contains a Python script demonstrating the use of Convolutional Neural Networks (CNN) for image classification, specifically utilizing the CIFAR-10 dataset. The project aims to showcase the performance benefits of GPU acceleration in image analysis tasks.

## Project Overview

The script implements a basic CNN model using TensorFlow and Keras. It is designed to classify images from the CIFAR-10 dataset, which includes 60,000 32x32 color images across 10 different classes. The focus is on measuring the execution time and accuracy of the model, highlighting the efficiency of GPU-accelerated computing in image processing tasks.

### Key Features

- **CNN Implementation:** A simple yet effective Convolutional Neural Network model suitable for image classification tasks.
- **GPU Acceleration:** Utilizes TensorFlow's GPU capabilities for faster computation.
- **Performance Metrics:** Measures execution time and accuracy to evaluate the model's performance.

## Prerequisites

Before running the script, ensure you have the following:

- Python 3.x
- TensorFlow 2.x (with GPU support if available)
- A suitable Python environment (like Anaconda)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/michaelkhany/SimpleMLModels.git
cd your-repo-name
```

## Usage
Run the script using a Python interpreter:

```
python cnn_cifar10.py
```

## Results
After execution, the script will output:

- Execution Time: The total time taken by the model to train on the CIFAR-10 dataset.
- Accuracy: The classification accuracy of the model on the test dataset.

## Contributing
Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License
This project is licensed under the MIT License.


### Notes:
- This script is a basic template. Depending on the specifics of your project, you might want to add or modify sections. For example, if you have specific installation instructions or dependencies, feel free to include those.
- You can enhance the README with images, badges, or more detailed instructions as needed after updating this template.


