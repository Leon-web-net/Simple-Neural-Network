// Stats.hpp
#ifndef STATS_HPP          // Include guard starts
#define STATS_HPP

#include <array>          
#include <iostream>
#include <cmath>

class Stats {
public:
    Stats();               // Constructor declaration
    ~Stats();              // Destructor declaration

    template<typename T, size_t N>
    double mean(const std::array<T,N> &arr);

    template<typename T, size_t N>
    double variance(const std::array<T,N> &arr, bool isSample = false);

    template<typename T, size_t N>
    double stdev(const std::array<T,N> &arr, bool isSample = false);
};

// Template implementations — must be in the header

template<typename T, size_t N>
double Stats::mean(const std::array<T,N> &arr) {
    T total{};
    for (size_t i = 0; i < arr.size(); i++) {
        total += arr[i];
    }
    return static_cast<double>(total) / N;
}

template<typename T, size_t N>
double Stats::variance(const std::array<T,N> &arr, bool isSample) {
    int n = N;
    double mean_ = mean(arr);
    double sum_diff_sqr{};
    for (size_t i = 0; i < arr.size(); i++) {
        sum_diff_sqr += (arr[i] - mean_) * (arr[i] - mean_);
    }
    if (isSample && n > 1) n -= 1;
    return static_cast<double>(sum_diff_sqr) / n;
}

template<typename T, size_t N>
double Stats::stdev(const std::array<T,N> &arr, bool isSample) {
    return std::sqrt(variance(arr, isSample));
}

Stats::Stats() {
        std::cout << "Maths Constructor called..." << std::endl;
    }

Stats::~Stats() {
        std::cout<<std::endl;
        std::cout << "\nMaths Deconstructor called..." << std::endl;
    }

#endif // STATS_HPP
