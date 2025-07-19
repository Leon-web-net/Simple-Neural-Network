#pragma once
#include <Eigen/Dense>
#include <filesystem>

struct ForwardResult {
    Eigen::MatrixXd Z1;
    Eigen::MatrixXd A1;
    Eigen::MatrixXd Z2;
    Eigen::MatrixXd A2;
};

struct BackwardResult{
    Eigen::MatrixXd dW1;
    Eigen::MatrixXd db1;
    Eigen::MatrixXd dW2;
    Eigen::MatrixXd db2;
};

class NeuralNetwork {
public:
    NeuralNetwork(const unsigned int input_size, const unsigned int hidden_size, 
        const unsigned int output_size, const double learning_rate,
        const std::filesystem::path& load_path = "");

    // Forward pass returns all intermediates in a struct
    ForwardResult forward(const Eigen::MatrixXd& X) const;
    
    BackwardResult backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, 
        const ForwardResult& forward_res) const;

    void updateParameters(const BackwardResult& grad);

    Eigen::RowVectorXi getPredictions(const Eigen::MatrixXd& A2);

    double NeuralNetwork::getAccuracy(const Eigen::RowVectorXi& predictions, 
    const Eigen::RowVectorXi& true_labels);

    // Getters (optional) for weights/biases
    const Eigen::MatrixXd& getW1() const { return W1; }
    const Eigen::VectorXd& getb1() const { return b1; }
    const Eigen::MatrixXd& getW2() const { return W2; }
    const Eigen::VectorXd& getb2() const { return b2; }

    void saveModel(const std::string& SAVE_PATH);
    void NeuralNetwork::UpdateLearningRate(const double factor);

private:
    Eigen::MatrixXd W1, W2;
    Eigen::VectorXd b1, b2;

    double learning_rate; // non const if decay or schedule
    const unsigned int input_size;
    const unsigned int hidden_size;
    const unsigned int output_size;

    static Eigen::MatrixXd ReLU(const Eigen::MatrixXd& x);
    static Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);
    static Eigen::MatrixXd YlabelMatrixIdx(const Eigen::MatrixXd& x, unsigned int num_classes);
};
