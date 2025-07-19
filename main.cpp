#include "./includes/NeuralNetwork.hpp"
#include "./includes/Model_IO.hpp"
#include <iostream>
#include <fstream>
#include <filesystem> 
#include <string>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <iomanip>


// ./Debug/main.exe
// cmake --build . --config Debug

using namespace std;
using Eigen::MatrixXd;

namespace {
    const string TRAIN_PATH = "./MNIST_handwritten_dataset/train.csv";
    // const string TRAIN_PATH = "./mnist_mock_train.csv";
    // const string TEST_PATH = "./MNIST_handwritten_dataset/test.csv";
    const string TEST_PATH = "./mnist_mock_test.csv";
    const string MODEL_DIR = "./model_params/";
    const float VALIDATION_SPLIT = 0.2f;

    const int INPUT_SIZE = 784;
    const int HIDDEN_SIZE = 10;
    const int OUTPUT_SIZE = 10;
    const int EPOCHS = 300;
    const double INITIAL_LR = 0.1;

    const int LR_DECAY_STEP = 50;
    const double LR_DECAY_FACTOR = 0.75;
    const int LR_DECAY_START = 50;
}


bool loadAndSplitData(MatrixXd& X_train, MatrixXd& Y_train, MatrixXd& X_dev, Eigen::RowVectorXi& Y_dev) {
    MatrixXd raw_data;
    if (csvRead(raw_data, TRAIN_PATH, true) != 0) {
        cerr << "Failed to load data from " << TRAIN_PATH << endl;
        return false;
    }

    raw_data.transposeInPlace(); // shape: (785 x m)
    int total = static_cast<int>(raw_data.cols());
    int dev_count = static_cast<int>(VALIDATION_SPLIT * total);
    int train_count = total - dev_count;

    Y_dev = raw_data.block(0, 0, 1, dev_count).cast<int>();
    X_dev = raw_data.block(1, 0, INPUT_SIZE, dev_count) / 255.0;

    Y_train = raw_data.block(0, dev_count, 1, train_count);
    X_train = raw_data.block(1, dev_count, INPUT_SIZE, train_count) / 255.0;

    return true;
}

void trainModel(NeuralNetwork& nn, const MatrixXd& X_train, const MatrixXd& Y_train,
                const MatrixXd& X_dev, const Eigen::RowVectorXi& Y_dev) {
    
    cout<<"Training started: \nEpochs "<<EPOCHS<<endl;
    cout<<"Learning Rate "<<INITIAL_LR<<endl;
    cout<<"Learning Rate Decay "<<LR_DECAY_FACTOR<<endl;
    cout<<"Learning Rate Step "<<LR_DECAY_STEP<<endl;
    cout<<"Learning Rate decay start "<<LR_DECAY_START<<endl;

    double best_accuracy = 0.0;
                    
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        ForwardResult forward_res = nn.forward(X_train);
        BackwardResult gradients = nn.backward(X_train, Y_train, forward_res);
        nn.updateParameters(gradients);

        if (epoch % 5 == 0) {
            auto preds = nn.getPredictions(nn.forward(X_dev).A2);
            double acc = nn.getAccuracy(preds, Y_dev);
            cout << "Epoch " << epoch << " - Accuracy: " << fixed << setprecision(2) << acc * 100 << "%\n";

            if (acc > best_accuracy) {
                best_accuracy = acc;
                nn.saveModel(MODEL_DIR);
            }
        }

        if (epoch >= LR_DECAY_START && epoch % LR_DECAY_STEP == 0) {
            nn.UpdateLearningRate(LR_DECAY_FACTOR);
            cout << "Learning rate decayed at epoch " << epoch << endl;
        }
    }
}

bool loadTestData(Eigen::MatrixXd& X_test) {
    MatrixXd raw_test_data;
    if (csvRead(raw_test_data, TEST_PATH, true) != 0) {
        std::cerr << "Failed to load test data from " << TEST_PATH << std::endl;
        return false;
    }

    raw_test_data.transposeInPlace(); // shape: (784 x m)
    X_test = raw_test_data / 255.0;

    return true;
}


std::vector<int> testModel(NeuralNetwork& nn, const Eigen::MatrixXd& images) {
    ForwardResult forward_res = nn.forward(images);
    Eigen::RowVectorXi preds = nn.getPredictions(forward_res.A2);

    // Convert Eigen::RowVectorXi to std::vector<int>
    std::vector<int> predictions(preds.data(), preds.data() + preds.size());
    return predictions;
}

int main() {
    MatrixXd X_train, Y_train, X_dev;
    Eigen::RowVectorXi Y_dev;

    // if (!loadAndSplitData(X_train, Y_train, X_dev, Y_dev)) {
    //     return EXIT_FAILURE;
    // }else{
    //     cout<<"Dataset successfully loaded"<<endl;
    // }


    
    NeuralNetwork nn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, INITIAL_LR, MODEL_DIR);
    cout<<"Neural Network Initialised"<<endl;

    // trainModel(nn, X_train, Y_train, X_dev, Y_dev);
    MatrixXd X_test;


    if (loadTestData(X_test)) {
        std::vector<int> predictions = testModel(nn, X_test);

        for (int i = 0; i < std::min(10, (int)predictions.size()); ++i) {
            std::cout << "Prediction[" << i << "] = " << predictions[i] << std::endl;
        }
    }else{
        return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}