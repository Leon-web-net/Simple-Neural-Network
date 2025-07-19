#pragma once
#include <string>
#include <Eigen/Dense>


int csvRead(Eigen::MatrixXd& outputMatrix, const std::string& fileName,
     const bool header = false,const std::streamsize dPrec=6);

void saveMatrixToCSV(const Eigen::MatrixXd& mat, const std::string& filename);
Eigen::MatrixXd loadMatrixFromCSV(const std::string& filename);

void saveMatrixBinary(const Eigen::MatrixXd& mat, const std::string& filename);
Eigen::MatrixXd loadMatrixBinary(const std::string& filename);


