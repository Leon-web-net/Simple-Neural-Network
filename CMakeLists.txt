cmake_minimum_required(VERSION 3.10)
project(MNIST_NeuralNet)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Eigen headers (adjust this path to where your Eigen is!)
include_directories("C:\\Program Files\\CPP_LIB\\eigen-3.4.0")

# Include your headers
include_directories("${CMAKE_SOURCE_DIR}/includes")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "C:\\Users\\letio\\OneDrive\\C++ Projects\\RandoCpp\\NN_CPP\\")

# Add the executable and source files
add_executable(main
    main.cpp
    includes/NeuralNetwork.cpp
    includes/Model_IO.cpp
)
