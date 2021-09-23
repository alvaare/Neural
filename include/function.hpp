#pragma once
#include <string>

enum functions {
    RELU,
    UNKNOWN
};

class Function {
    private:
        std::string name;
        functions id;
    public:
        Function(functions);
        double f(double);
        double df(double);
};