#include "NeuralNetwork.hpp"
#include "Model_IO.hpp"
#include <random>
#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <iomanip> 
#include <omp.h>

NeuralNetwork::NeuralNetwork(const unsigned int input_size_, const unsigned int hidden1_size_, 
            const unsigned int hidden2_size_, const unsigned output_size_, const double learning_rate_, 
            const std::filesystem::path& load_path)
            : input_size(input_size_), hidden1_size(hidden1_size_),hidden2_size(hidden2_size_),
             output_size(output_size_),learning_rate(learning_rate_){

    if (!load_path.empty()) {
         try {
            std::cout<<"Loading weights and biases... "<<std::endl;
            std::filesystem::path path(load_path);
            W1 = loadMatrixFromCSV((path / "model_W1.csv").string());
            b1 = loadMatrixFromCSV((path / "model_b1.csv").string());
            W2 = loadMatrixFromCSV((path / "model_W2.csv").string());
            b2 = loadMatrixFromCSV((path / "model_b2.csv").string());
            W3 = loadMatrixFromCSV((path / "model_W3.csv").string());
            b3 = loadMatrixFromCSV((path / "model_b3.csv").string());

            std::cout << "Model loaded from: " << path << std::endl;
            return;
        } catch (const std::exception& e) {
        std::cerr << "Failed to load model from path: " << load_path << ".\n";
        std::cerr << "Error: " << e.what() << std::endl;
        }
    }else{
                            
        std::random_device rd;
        std::mt19937 rng(rd());

        auto rand_matrix_he = [&](Eigen::MatrixXd& matrix, unsigned int rows, unsigned int cols) {
            double stddev = std::sqrt(2.0 / cols); 
            std::normal_distribution<> dis(0.0, stddev);
            matrix = Eigen::MatrixXd::NullaryExpr(rows, cols, [&]() { return dis(rng); });
        };

        // Xavier initialization for softmax/output layer
        auto rand_matrix_xavier = [&](Eigen::MatrixXd& matrix, unsigned int rows, unsigned int cols) {
            double limit = std::sqrt(6.0 / (rows + cols));
            std::uniform_real_distribution<> dis(-limit, limit);
            matrix = Eigen::MatrixXd::NullaryExpr(rows, cols, [&]() { return dis(rng); });
        };

        // Biases initialized to zero (common practice)
        auto rand_vector_zero = [&](Eigen::VectorXd& vec, unsigned int size) {
            vec = Eigen::VectorXd::Zero(size);
        };

        // Apply
        rand_matrix_he(W1, hidden1_size, input_size);
        rand_vector_zero(b1, hidden1_size);

        rand_matrix_he(W2, hidden2_size, hidden1_size);
        rand_vector_zero(b2, hidden2_size);

        rand_matrix_xavier(W3, output_size, hidden2_size);
        rand_vector_zero(b3, output_size);
    }
    return;
}

ForwardResult NeuralNetwork::forward(const Eigen::MatrixXd& X) const {
    ForwardResult res;

    // First hidden layer
    res.Z1 = W1 * X;
    res.Z1.colwise() += b1;
    res.A1 = ReLU(res.Z1);

    // Second hidden layer
    res.Z2 = W2 * res.A1;
    res.Z2.colwise() += b2;
    res.A2 = ReLU(res.Z2);

    // Output layer
    res.Z3 = W3 * res.A2;
    res.Z3.colwise() +=b3;
    res.A3 = softmax(res.Z3);
        
    return res;
}

Eigen::MatrixXd NeuralNetwork::ReLU(const Eigen::MatrixXd& x) {
    return x.cwiseMax(0.0);
}

Eigen::MatrixXd NeuralNetwork::softmax(const Eigen::MatrixXd& x) {

    Eigen::RowVectorXd max_per_col = x.colwise().maxCoeff();  // (1 x cols)
    Eigen::MatrixXd shifted = x.rowwise() - max_per_col;      // broadcast rowwise
    Eigen::MatrixXd exp_x = shifted.array().exp();
    Eigen::RowVectorXd sums = exp_x.colwise().sum();
    return exp_x.array().rowwise() / sums.array();      
}

BackwardResult NeuralNetwork::backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const ForwardResult& forward_res)const{
    BackwardResult grad;

    const auto& Z1 = forward_res.Z1; // shape: (hidden1_size x m)
    const auto& A1 = forward_res.A1; // shape: (hidden1_size x m)
    const auto& Z2 = forward_res.Z2; // shape: (hidden2_size x m)
    const auto& A2 = forward_res.A2; // shape: (hidden2_size x m)
    const auto& Z3 = forward_res.Z3; // shape: (output_size x m)
    const auto& A3 = forward_res.A3; // shape: (output_size x m)

    const auto m = Y.cols();  // number of samples
    const double inv_m = 1.0/m;

    Eigen::MatrixXd one_hot_Y = YlabelMatrixIdx(Y,output_size); // (shape: output_size x m) 
    
    // Output layer gradient
    Eigen::MatrixXd dZ3 = A3 - one_hot_Y; // shape: (output_size x m)
    grad.dW3 = inv_m * dZ3 * A2.transpose();   // shape: (output_size x hidden_size)
    grad.db3 = inv_m * dZ3.rowwise().sum();  // shape: (output_size x 1)

    
    // Second hidden layer gradients
    Eigen::MatrixXd relu_deriv2 = (Z2.array()>0.0).cast<double>(); // shape: (hidden2_size x m)
    Eigen::MatrixXd dZ2 = (W3.transpose()*dZ3).cwiseProduct(relu_deriv2); // shape: (hidden2_size x m)
    grad.dW2 = inv_m *dZ2 * A1.transpose();  // (hidden2_size x hidden1_size)
    grad.db2 = inv_m *dZ2.rowwise().sum(); // (hidden2_size x 1)

    // Derivative of ReLU
    Eigen::MatrixXd relu_deriv1 = (Z1.array() > 0).cast<double>(); // shape: (hidden1_size x m)
    Eigen::MatrixXd dZ1 = (W2.transpose() * dZ2).cwiseProduct(relu_deriv1); // shape: (hidden1_size x m)
    grad.dW1 = inv_m * dZ1 * X.transpose(); // (hidden1_size x input_size)
    grad.db1 = inv_m * dZ1.rowwise().sum(); // (hidden1_size x 1)

    return grad;
}

Eigen::MatrixXd NeuralNetwork::YlabelMatrixIdx(const Eigen::MatrixXd& Y, unsigned int num_classes){
    // Pseudo one hot I guess
    assert(Y.rows() == 1 && "Y should be a row vector");
    
    auto m = Y.cols();
    Eigen::MatrixXd one_hot = Eigen::MatrixXd::Zero(num_classes,m);

    for (int i = 0; i < m; ++i) {
        int label = static_cast<int>(Y(0, i));
        if (label >= 0 && label < static_cast<int>(num_classes)) {
            one_hot(label, i) = 1.0;
        } else {
            std::cerr << "Warning: label " << label << " is out of bounds for one-hot encoding.\n";
        }
    }

    return one_hot;
}

void NeuralNetwork::updateParameters(const BackwardResult& grad){
    double LR = learning_rate;
    
    W1.noalias() = W1 - LR * grad.dW1;
    b1.noalias() = b1 - LR * grad.db1;
    W2.noalias() = W2 - LR * grad.dW2;
    b2.noalias() = b2 - LR * grad.db2;
    W3.noalias() = W3 - LR * grad.dW3;
    b3.noalias() = b3 - LR * grad.db3;

    return;
}

Eigen::RowVectorXi NeuralNetwork::getPredictions(const Eigen::MatrixXd& A3){
    Eigen::RowVectorXi predictions(A3.cols());

    #pragma omp parallel for
    for(int i =0; i<A3.cols();i++){
        A3.col(i).maxCoeff(&predictions(i));
    }

    return predictions;
}

double NeuralNetwork::getAccuracy(const Eigen::RowVectorXi& predictions, 
    const Eigen::RowVectorXi& true_labels){
        
    assert(predictions.size() == true_labels.size() && "Size mismatch");

    Eigen::Array<bool,1,Eigen::Dynamic> correct = (predictions.array()==true_labels.array());
    const auto num_correct = correct.count();

    return static_cast<double>(num_correct)/predictions.size();
    }

void NeuralNetwork::UpdateLearningRate(const double factor){
    assert(factor <= 1 && factor > 0);
    learning_rate *= factor;

     std::cout << std::fixed << std::setprecision(8);
    std::cout<<"Learning Rate updated by a factor "<<factor<<std::endl;
    std::cout<<"New Learning Rate "<<learning_rate<<std::endl;
    
    return;
}

void NeuralNetwork::saveModel(const std::string& SAVE_PATH){
    if (!SAVE_PATH.empty()) {
        std::filesystem::path save_dir(SAVE_PATH);

        if (!std::filesystem::exists(save_dir)) {
            if (!std::filesystem::create_directories(save_dir)) {
                std::cerr << "Failed to create directory: " << SAVE_PATH << std::endl;
                return;
            }
        }

        std::vector<std::string> filenames = {
            "model_W1.csv", "model_b1.csv", "model_W2.csv", "model_b2.csv",
            "model_W3.csv", "model_b3.csv"
        };

        for (const auto& fname : filenames) {
            std::filesystem::path full_path = save_dir / fname;
            // if (std::filesystem::exists(full_path)) {
            //     std::cout << "Overwriting existing file: " << full_path << std::endl;
            // }
        }

        saveMatrixToCSV(W1, (save_dir / "model_W1.csv").string());
        saveMatrixToCSV(b1, (save_dir / "model_b1.csv").string());
        saveMatrixToCSV(W2, (save_dir / "model_W2.csv").string());
        saveMatrixToCSV(b2, (save_dir / "model_b2.csv").string());
        saveMatrixToCSV(W3, (save_dir / "model_W3.csv").string());
        saveMatrixToCSV(b3, (save_dir / "model_b3.csv").string());

        std::cout << "Model saved to: " << SAVE_PATH << std::endl;
    }
}