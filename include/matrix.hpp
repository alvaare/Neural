#pragma once
#include <string>

class Matrix {
    private:
        int height;
        int width;
        double* data;
    public:
        Matrix(int, int, std::string);
        Matrix();
        ~Matrix();
        void print();

        double get(int, int);
        void set(int, int, double);
};