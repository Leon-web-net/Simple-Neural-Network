# 🧠 NN_CPP - Neural Network in C++ for MNIST Classification

A simple feedforward neural network implemented from scratch in **C++** using **Eigen** for matrix operations. The model is trained on the **MNIST** dataset for handwritten digit classification.

---

## 🚀 Features

- Feedforward neural network with:
  - One hidden layer
  - Sigmoid activation
  - Softmax output layer
- Multithreaded prediction using OpenMP
- Custom training loop with:
  - Forward and backward propagation
  - Gradient descent
  - Validation split
- Model save/load from `.csv` files
- MNIST data preprocessing
- CMake-based build system

## 🔧 Dependencies

- [Eigen 3](https://eigen.tuxfamily.org) (header-only)
- C++17 or higher
- CMake ≥ 3.10
- OpenMP (for multithreading support)

---

## Results

Trained on MNIST with the following configuration:

- Input size: 784
- Hidden layer size: 128
- Output size: 10
- Epochs: 200
- Initial learning rate: 0.1
- Learning rate decay: 0.75 every 20 epochs (starting at epoch 40)
- Validation split: 20%
- Multithreading enabled with OpenMP (16 threads)

**Final validation accuracy:** 81.94%

---

## 🛠️ Build Instructions

```bash
# Clone the repo
git clone https://github.com/Leon-web-net/Simple-Neural-Network.git
cd NN_CPP

# Create build directory
mkdir build && cd build

# Run CMake and build
cmake ..
cmake --build . --config Release
```

## References & Credits

This project was inspired by Samson Zhang neural network tutorial in python
[Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://youtu.be/w8yWXqWQYmU?si=z3q99MKA7Ig4fPc1)

[MNIST Dataset (Kaggle)](https://www.kaggle.com/competitions/digit-recognizer/overview)

[Eigen Library](https://eigen.tuxfamily.org/index.php?title=Main_Page)

[csvRead function concept from this C++ tutorial ](https://youtu.be/m118or4f0FE?si=Jhx_WEh-DisEJbiH)
