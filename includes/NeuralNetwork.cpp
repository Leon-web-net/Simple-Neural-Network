#include "NeuralNetwork.hpp"
#include "Model_IO.hpp"
#include <random>
#include <Eigen/Dense>
#include <filesystem>
#include <iostream>

NeuralNetwork::NeuralNetwork(const unsigned int input_size_, const unsigned int hidden_size_, 
            const unsigned output_size_, const double learning_rate_, const std::filesystem::path& load_path)
            : input_size(input_size_), hidden_size(hidden_size_), output_size(output_size_), 
            learning_rate(learning_rate_){

    if (!load_path.empty()) {
         try {
            std::cout<<"Loading weights and biases... "<<std::endl;
            std::filesystem::path path(load_path);
            W1 = loadMatrixFromCSV((path / "model_W1.csv").string());
            b1 = loadMatrixFromCSV((path / "model_b1.csv").string());
            W2 = loadMatrixFromCSV((path / "model_W2.csv").string());
            b2 = loadMatrixFromCSV((path / "model_b2.csv").string());

            std::cout << "Model loaded from: " << path << std::endl;
            return;
        } catch (const std::exception& e) {
        std::cerr << "Failed to load model from path: " << load_path << ".\n";
        std::cerr << "Error: " << e.what() << std::endl;
        }
    }else{
                            
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<> dis(-0.5, 0.5);

        auto rand_matrix = [&](Eigen::MatrixXd& matrix, unsigned int rows, unsigned int cols) {
            matrix = Eigen::MatrixXd::NullaryExpr(rows, cols, [&]() { return dis(rng); });
        };

        auto rand_vector = [&](Eigen::VectorXd& vec, unsigned int size) {
            vec = Eigen::VectorXd::NullaryExpr(size, [&]() { return dis(rng); });
        };

        rand_matrix(W1, hidden_size, input_size);
        rand_vector(b1, hidden_size);
        rand_matrix(W2, output_size, hidden_size);
        rand_vector(b2, output_size);
    
    }
    return;
}

ForwardResult NeuralNetwork::forward(const Eigen::MatrixXd& X) const {
    ForwardResult res;

    res.Z1 = W1 * X;

    res.Z1.colwise() += b1;

    res.A1 = ReLU(res.Z1);

    res.Z2 = W2 * res.A1;

    res.Z2.colwise() += b2;

    res.A2 = softmax(res.Z2);
        
    return res;
}

Eigen::MatrixXd NeuralNetwork::ReLU(const Eigen::MatrixXd& x) {
    return x.cwiseMax(0);
}

Eigen::MatrixXd NeuralNetwork::softmax(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd exp_x = x.array().exp();
    Eigen::VectorXd sums = exp_x.colwise().sum();

    return exp_x.array().rowwise() / sums.transpose().array();
    
}

BackwardResult NeuralNetwork::backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const ForwardResult& forward_res)const{
    BackwardResult grad;

    const auto& Z1 = forward_res.Z1; // shape: (hidden_size x m)
    const auto& A1 = forward_res.A1; // shape: (hidden_size x m)
    const auto& Z2 = forward_res.Z2; // shape: (output_size x m)
    const auto& A2 = forward_res.A2; // shape: (output_size x m)

    const auto m = Y.cols();  // number of samples

    Eigen::MatrixXd one_hot_Y = YlabelMatrixIdx(Y,output_size); // (shape: output_size x m) 
   
    Eigen::MatrixXd dZ2 = A2 - one_hot_Y; // shape: (output_size x m)

    grad.dW2 = (1.0 / m) * dZ2 * A1.transpose();   // shape: (output_size x hidden_size)

    grad.db2 = (1.0 / m) * dZ2.rowwise().sum();  // shape: (output_size x 1)

    // Derivative of ReLU
    Eigen::MatrixXd relu_deriv = (Z1.array() > 0).cast<double>(); // shape: (hidden_size x m)
   
    Eigen::MatrixXd dZ1 = (W2.transpose() * dZ2).cwiseProduct(relu_deriv); // shape: (hidden_size x m)
   

    grad.dW1 = (1.0 / m) * dZ1 * X.transpose(); // (hidden_size x input_size)

    grad.db1 = (1.0 / m) * dZ1.rowwise().sum(); // (hidden_size x 1)

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
    W1 = W1 - LR*grad.dW1;
    b1 = b1 - LR*grad.db1;
    W2 = W2 - LR*grad.dW2;
    b2 = b2 - LR*grad.db2;

    return;
}

Eigen::RowVectorXi NeuralNetwork::getPredictions(const Eigen::MatrixXd& A2){
    Eigen::RowVectorXi predictions(A2.cols());

    for(int i =0; i<A2.cols();i++){
        A2.col(i).maxCoeff(&predictions(i));
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
    assert(factor < 1 && factor > 0);
    learning_rate *= factor;
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
            "model_W1.csv", "model_b1.csv", "model_W2.csv", "model_b2.csv"
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

        std::cout << "Model saved to: " << SAVE_PATH << std::endl;
    }
}