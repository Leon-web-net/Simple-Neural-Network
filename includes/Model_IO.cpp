#include "Model_IO.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <Eigen/Dense>


int csvRead(Eigen::MatrixXd& outputMatrix, const std::string& fileName,
     const bool header,const std::streamsize dPrec){
    std::ifstream inputData;
    inputData.open(fileName);
    std::cout.precision(dPrec);
    if(!inputData){
        std::cout<<"Invalid input data"<<std::endl;
        return -1;
    }

    std::string fileline, filecell;
    unsigned int prevNumOfCols = 0, numOfRows =0, numOfCols = 0;
    
    if(header)getline(inputData,fileline);

    while(getline(inputData,fileline)){
        numOfCols = 0;
        std::stringstream linestream(fileline);  
        while(getline(linestream,filecell,',')){
            try{
                stod(filecell);
            }
            catch(...){
                std::cout<<"catch in first loop"<<std::endl;
                return -1;
            }
            numOfCols++;
        }
        if(numOfRows++ == 0)
            prevNumOfCols =numOfCols;
        if(prevNumOfCols != numOfCols){
            std::cout<<"number of columns inconsistent"<<std::endl;
            return -1;
        }
    }
    inputData.close();
    outputMatrix.resize(numOfRows,numOfCols);
    inputData.open(fileName);

    if(header)getline(inputData,fileline);
    
    numOfRows =0;
    while(getline(inputData,fileline)){
        numOfCols =0;
        std::stringstream linestream(fileline);
        while(getline(linestream,filecell,',')){
            outputMatrix(numOfRows,numOfCols++) = stod(filecell);
        }
        numOfRows++;
    }
    return 0;
}

void saveMatrixToCSV(const Eigen::MatrixXd& mat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for write: " << filename << std::endl;
        return;
    }

    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n");
    file << mat.format(CSVFormat);
    file.close();
}


Eigen::MatrixXd loadMatrixFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for read: " << filename << std::endl;
        return Eigen::MatrixXd();
    }

    std::vector<std::vector<double>> values;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string val;
        while (std::getline(ss, val, ',')) {
            row.push_back(std::stod(val));
        }
        values.push_back(row);
    }

    if (values.empty()) return Eigen::MatrixXd();

    Eigen::MatrixXd mat(values.size(), values[0].size());
    for (size_t i = 0; i < values.size(); ++i)
        mat.row(i) = Eigen::VectorXd::Map(values[i].data(), values[i].size());

    return mat;
}

void saveMatrixBinary(const Eigen::MatrixXd& mat, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (out.is_open()) {
        Eigen::Index rows = mat.rows(), cols = mat.cols();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char*>(mat.data()), sizeof(double) * rows * cols);
        out.close();
    }
}

Eigen::MatrixXd loadMatrixBinary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening file for binary read: " << filename << std::endl;
        return Eigen::MatrixXd();
    }

    Eigen::Index rows = 0, cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    Eigen::MatrixXd mat(rows, cols);
    in.read(reinterpret_cast<char*>(mat.data()), sizeof(double) * rows * cols);
    in.close();

    return mat;
}
