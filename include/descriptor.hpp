#pragma once
#include <string>
#include "function.hpp"

class Shape {
    private:
        int depth;
        int* shape;
    
    public:
        Shape();
        Shape(std::string);
        void print();
};

class Descriptor {
    private:
        Shape shape;
        std::string input_file;
        std::string output_file;
        double learning_rate;
        Function cost_function;
        Function activation_function;

    public:
        Descriptor(std::string descriptor_file);
        void print();

        int get_depth();
        int* get_shape();
        std::string get_input_file();
        std::string get_output_file();
        Function* get_cost_function();
        Function* get_activation_functio();


};