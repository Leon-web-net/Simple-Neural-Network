cmake_minimum_required(VERSION 3.10)
project(MNIST_NeuralNet)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Eigen headers (adjust this path to where your Eigen is!)
include_directories("C:\\Program Files\\CPP_LIB\\eigen-3.4.0")

# Include your headers
include_directories("${CMAKE_SOURCE_DIR}/includes")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "C:\\Users\\letio\\NN_CPP\\")

# Add the executable and source files
add_executable(main
    main.cpp
    includes/NeuralNetwork.cpp
    includes/Model_IO.cpp
)

#  Enable OpenMP
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()
