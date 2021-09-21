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

        int* get_shape() const;
        int get_depth() const;
        int get_size_layer(int) const;
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
        ~Descriptor();
        void print();

        const std::string get_input_file() const;
        const std::string get_output_file() const;
        //Function* get_cost_function();
        //Function* get_activation_functio();
        const Shape& get_shape() const;

};