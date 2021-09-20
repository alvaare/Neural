#include <fstream>
#include <iostream>
#include <sstream> 
#include "matrix.hpp"

Matrix::Matrix(int h, int w, std::string file) {
    height = h;
    width = w;
    data = new double[height*width];
    std::string line;
    std::ifstream myfile (file);
    int id_number = 0;
    while (getline(myfile, line)) {
        std::string value;
        std::stringstream myline(line);
        while (getline(myline, value, ' ')) {
            data[id_number] = stod(value);
            id_number++;
        }
    }
}

Matrix::Matrix() = default;

Matrix::~Matrix() {
    delete [] data;
}

void Matrix::print() {
    std::cout << "Printing Matrix:\n";
    std::cout << "Height: " << height << "\n";
    std::cout << "width: " << width << "\n";
    for (int i = 0; i < height*width; i++) {
        std::cout << data[i] << " ";
        if ((i-1) % width == 0) {
            std::cout << "\n";
        }
    } 
}

double Matrix::get(int n, int m) {
    return data[n*width+m];
}

void Matrix::set(int n, int m, double value) {
    data[n*width+m] = value;
}