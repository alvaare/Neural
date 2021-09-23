#include <iostream>
#include "function.hpp"

double Relu(double x) {
    return std::max(0.0, x);
}

double dRelu(double x) {
    return (x > 0)? 1 : 0;
}

Function::Function(functions f_token) {
    switch(f_token) {
        case RELU :
            name = "ReLU";
            id = RELU;
            break;
        case UNKNOWN :
            std::cout << "Unknown activation function\n";
    }
}

double Function::f(double x) {
    switch(id) {
        case RELU :
            return Relu(x);
    }
}

double Function::df(double x) {
    switch(id) {
        case RELU :
            return dRelu(x);
    }
}