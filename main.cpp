#include "./includes/NeuralNetwork.hpp"
#include "./includes/Model_IO.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <filesystem> 
#include <string>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <iomanip>
#include <chrono>



// ./Debug/main.exe
// cmake --build . --config Debug

using namespace std;
using Eigen::MatrixXd;

namespace {
    const string TRAIN_PATH = "./MNIST_handwritten_dataset/train.csv";
    // const string TRAIN_PATH = "./mnist_mock_train.csv";
    // const string TEST_PATH = "./MNIST_handwritten_dataset/test.csv";
    const string TEST_PATH = "./mnist_mock_test.csv";
    const string TEST_IMAGE_DIR = "./MNIST_images/";

    const string MODEL_DIR = "./model_params/";
    // const string LOAD_PATH = "./model_params/"
    const string LOAD_PATH = "";
    const float VALIDATION_SPLIT = 0.2f;

    const int INPUT_SIZE = 784;
    const int HIDDEN1_SIZE = 128;
    const int HIDDEN2_SIZE = 64;
    const int OUTPUT_SIZE = 10;
    const int EPOCHS = 25;
    const int BATCH_SIZE = 64; 
    const double INITIAL_LR = 0.05;

    const int LR_DECAY_STEP = 5;
    const double LR_DECAY_FACTOR = 0.9;
    const int LR_DECAY_START = 5;
    
    const bool IS_THREADS = true;
    const int NUM_THREADS = 16;
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
    int num_batches =  static_cast<int>(X_train.cols())/BATCH_SIZE;
    std::mt19937 rng(42);
    

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X_train.cols());
        perm.setIdentity();
        std::shuffle(perm.indices().data(),
        perm.indices().data()+ perm.indices().size(),
            rng);
        
        Eigen::MatrixXd X_shuffled = X_train * perm;
        Eigen::MatrixXd Y_shuffled = Y_train * perm;

        for(int b = 0; b<num_batches; ++b){

            int start =  b* BATCH_SIZE;
            int count = std::min(BATCH_SIZE, static_cast<int>(X_shuffled.cols())- start);

            Eigen::MatrixXd X_batch = X_shuffled.middleCols(start,count);
            Eigen::MatrixXd Y_batch = Y_shuffled.middleCols(start,count);

            ForwardResult forward_res = nn.forward(X_batch);
            BackwardResult gradients = nn.backward(X_batch, Y_batch, forward_res);
            nn.updateParameters(gradients);
        }

        if (epoch % 5 == 0) {
            auto preds = nn.getPredictions(nn.forward(X_dev).A3);
            double acc = nn.getAccuracy(preds, Y_dev);
            cout << "Epoch " << epoch << " - Accuracy: " <<
                 fixed << setprecision(2) << acc * 100 << "%\n";

            if (acc > best_accuracy) {
                best_accuracy = acc;
                nn.saveModel(MODEL_DIR);
                // cout<<"Model saved "<<MODEL_DIR<<endl;
            }
        }

        if (epoch >= LR_DECAY_START && epoch % LR_DECAY_STEP == 0) {
            nn.UpdateLearningRate(LR_DECAY_FACTOR);
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

bool loadTestImagesOpenCV(Eigen::MatrixXd& X_test, std::vector<std::string>& image_paths) {
    namespace fs = std::filesystem;

    // Collect PNG files
    try {
        for (const auto& entry : fs::directory_iterator(TEST_IMAGE_DIR)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                image_paths.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
        return false;
    }

    if (image_paths.empty()) {
        std::cerr << "No PNG files found in " << TEST_IMAGE_DIR << std::endl;
        return false;
    }

    std::sort(image_paths.begin(), image_paths.end());
    const int num_images = static_cast<int>(image_paths.size());
    X_test.resize(784, num_images);

    for (int i = 0; i < num_images; ++i) {
        cv::Mat img = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << image_paths[i] << std::endl;
            return false;
        }

        if (img.rows != 28 || img.cols != 28)
            cv::resize(img, img, cv::Size(28, 28));

        img.convertTo(img, CV_64F, 1.0/255.0);

        // Flatten image using reshape
        Eigen::Map<Eigen::VectorXd> col(img.ptr<double>(), 28*28);
        X_test.col(i) = col;
    }

    std::cout << "Loaded " << num_images << " test images." << std::endl;
    return true;
}

void displayImagesWithPredictions(const std::vector<int>& predictions, 
                                  const std::vector<std::string>& image_paths) {
    
    int max_show = std::min(10, static_cast<int>(predictions.size()));

    for (int i = 0; i < max_show; ++i) {
        cv::Mat img = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << image_paths[i] << std::endl;
            continue;
        }

        // Resize image for better visibility
        cv::Mat disp;
        cv::resize(img, disp, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);

        // Add predicted label text
        std::string text = "Predicted: " + std::to_string(predictions[i]);
        cv::putText(disp, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(128), 2);

        // Show in **same window** for all images
        cv::imshow("Prediction", disp);

        // Wait until a key is pressed
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
}

// std::vector<int> testModel(NeuralNetwork& nn, const Eigen::MatrixXd& X_test) {
//     Eigen::RowVectorXi preds = nn.getPredictions(nn.forward(X_test).A3);
//     return { preds.data(), preds.data() + preds.size() };
// }

std::vector<int> testModel(NeuralNetwork& nn, const Eigen::MatrixXd& images) {
    ForwardResult forward_res = nn.forward(images);
    Eigen::RowVectorXi preds = nn.getPredictions(forward_res.A3);

    std::vector<int> predictions(preds.data(), preds.data() + preds.size());
    return predictions;
}

int main() {

    cout<<"Running main..."<<endl;
    const bool train = false;
    const bool test = true;
    
    if(IS_THREADS){
        Eigen::initParallel();
        int num_threads = std::thread::hardware_concurrency();
        cout<<"Number of threads available "<<num_threads<<endl;
        if(num_threads == 0) num_threads = 1;  // fallback
        if(num_threads>=NUM_THREADS) num_threads = NUM_THREADS; 
        Eigen::setNbThreads(num_threads);
        cout<<"Using "<<NUM_THREADS<<" logic cores"<<endl;
    }

    MatrixXd X_train, Y_train, X_dev;
    Eigen::RowVectorXi Y_dev;
   


    if(train){
        NeuralNetwork nn(INPUT_SIZE, HIDDEN1_SIZE,HIDDEN2_SIZE, OUTPUT_SIZE, INITIAL_LR,LOAD_PATH);
        if (!loadAndSplitData(X_train, Y_train, X_dev, Y_dev)) {
            return EXIT_FAILURE;
        }else{
            cout<<"Dataset successfully loaded"<<endl;
        }

    cout<<"Neural Network Initialised"<<endl;
    
    auto start = std::chrono::high_resolution_clock::now();

    trainModel(nn, X_train, Y_train, X_dev, Y_dev);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds\n";
    }

    cv::Mat test_mat = cv::Mat::zeros(10, 10, CV_8UC1);
    if (test_mat.empty()) {
        std::cerr << "OpenCV not working!" << std::endl;
        return -1;
    }
    std::cout << "OpenCV is working" << std::endl;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    if(test){
        NeuralNetwork nn(INPUT_SIZE, HIDDEN1_SIZE,HIDDEN2_SIZE, OUTPUT_SIZE, INITIAL_LR,MODEL_DIR);
        MatrixXd X_test;
        std::vector<std::string> image_names;

        if (loadTestImagesOpenCV(X_test, image_names)) {
            std::vector<int> predictions = testModel(nn, X_test);
            
            // Display images with predictions
            displayImagesWithPredictions(predictions, image_names);
            
        } else {
            return EXIT_FAILURE;
        }
    }


    return EXIT_SUCCESS;
}