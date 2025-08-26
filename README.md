# 🧠 NN_CPP - Neural Network in C++ for MNIST Classification

A feedforward neural network implemented from scratch in **C++** using **Eigen** for matrix operations. The model is trained on the **MNIST** dataset for handwritten digit classification.

---

## 🚀 Features

- Feedforward neural network with:
  - **Two hidden layers** (128 → 64)
  - Sigmoid activation
  - Softmax output layer
- **Mini-batch gradient descent**
- **Multithreaded prediction** using OpenMP
- Custom training loop with:
  - Forward and backward propagation
  - Gradient descent with learning rate decay
  - Validation split
- Model save/load from `.csv` files
- MNIST data preprocessing
- **OpenCV integration** to visualize test images with predicted labels
- CMake-based build system

## 🔧 Dependencies

- [Eigen 3](https://eigen.tuxfamily.org) (header-only)
- OpenCV (for image visualization)
- C++17 or higher
- CMake ≥ 3.10
- OpenMP (for multithreading support)

---

## Results

Trained on MNIST with the following configuration:

- Input size: 784
- Hidden layers: 128 → 64
- Output size: 10
- Epochs: 25
- Mini-batch size: 64
- Initial learning rate: 0.05
- Learning rate decay: 0.9 every 5 epochs (starting at epoch 5)
- Validation split: 20%
- Multithreading enabled with OpenMP (16 threads)
- OpenCV for visualizing predictions

**Final validation accuracy:** 97.39%

---

## 🛠️ Build Instructions

1. Clone the repo
   git clone https://github.com/Leon-web-net/Simple-Neural-Network.git
   cd NN_CPP

2. Install dependencies

- Eigen: unzip somewhere like C:/CPP_LIB/eigen-3.4.0
- OpenCV: prebuilt binaries in C:/CPP_LIB/opencv

3. Ensure OpenCV DLLs are in your PATH
   Example: add C:/CPP_LIB/opencv/build/x64/vc16/bin to system PATH

4. Create build directory
   mkdir build
   cd build

5. Run CMake to configure the project
   cmake .. -G "MinGW Makefiles" # or "Visual Studio 16 2019" if using MSVC

6. Build the project
   cmake --build . --config Release

---

## Future Work

- **Expand network architecture:** Add more hidden layers and experiment with different activation functions to improve accuracy and model capacity.
- **Visualisation tools:** Integrate visualisations to display CSV image data and training metrics (e.g., loss and accuracy curves) for better insight into model performance.
- **Optimisation:** Implement additional training techniques such as momentum, Adam optimiser, or batch normalisation.

---

## References & Credits

This project was inspired by Samson Zhang neural network tutorial in python
[Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://youtu.be/w8yWXqWQYmU?si=z3q99MKA7Ig4fPc1)

[MNIST Dataset (Kaggle)](https://www.kaggle.com/competitions/digit-recognizer/overview)

[Eigen Library](https://eigen.tuxfamily.org/index.php?title=Main_Page)

[csvRead function concept from this C++ tutorial ](https://youtu.be/m118or4f0FE?si=Jhx_WEh-DisEJbiH)

---

## Current update included:

- Mentioned **two hidden layers**.
- Highlighted **mini-batch gradient descent**.
- Added **OpenCV visualization**.
- Updated **accuracy to 97.39%** at 25 epochs.
- Reflected hyperparameters (`BATCH_SIZE`, `EPOCHS`, `LR`).
