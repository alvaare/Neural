#pragma once
#include <string>

class Function {
    private:
        std::string name;
    public:
        double f(double);
        double df(double);
};