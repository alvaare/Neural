#include <iostream>
#include <fstream>
#include <assert.h>
#include "descriptor.hpp"

enum tokens {
    INPUT_LINE,
    OUTPUT_LINE,
    SHAPE,
    ERROR
};

tokens resolve_token(std::string token) {
    if (token == "INPUT_FILE") return INPUT_LINE;
    if (token == "OUTPUT_FILE") return OUTPUT_LINE;
    if (token == "SHAPE") return SHAPE;
    return ERROR;
}

tokens get_token(std::string line) {
    std::string token;
    for (char c : line) {
        if (c == ' ') {
            return resolve_token(token);
        }
        token.push_back(c);
    }
    return ERROR;
}

std::string get_description(std::string line) {
    for (int i = 0; i < (int)line.length(); i++) {
        if (line[i] == '=') {
            return line.substr(i+2, line.length()-(i+2));
        }
    }
    throw ("There is no '=' in:\n" + line);
}

int nb_commas(std::string description) {
    int res = 0;
    for (char c : description) {
        if (c == ',') {
            res++;
        }
    }
    return res;
}

void fill_shape(int* shape, std::string shape_description, int depth) {
    std::string nb_neurons;
    int id_layer = 0;
    for (char c : shape_description) {
        switch(c) {
            case ' ':
                break;
            case ',':
                shape[id_layer] = stoi(nb_neurons);
                id_layer++;
                nb_neurons.clear();
                break;
            default:
                nb_neurons.push_back(c);
        }
    }
    shape[id_layer] = stoi(nb_neurons);
}

Shape::Shape(std::string shape_description) {
    depth = nb_commas(shape_description) + 1;
    shape = new int[depth];
    fill_shape(shape, shape_description, depth);
}

Shape::Shape() {
    depth = 0;
    shape = NULL;
}

void Shape::print() {
    std::cout << "Printing Shape:\n";
    std::cout << "Depth: " << depth << "\n";
    std::cout << "Shape: ";
    for (int id_layer = 0; id_layer < depth; id_layer++) {
        std::cout << shape[id_layer] << " ";
    }
    std::cout << "\n";
}

Descriptor::Descriptor(std::string descriptor_file) {
    std::string line;
    std::ifstream myfile (descriptor_file);
    while (getline(myfile, line)) {
        tokens token = get_token(line);
        std::string description = get_description(line);
        switch(token) {
            case INPUT_LINE:
                input_file = description;
                break;
            case OUTPUT_LINE:
                output_file = description;
                break;
            case SHAPE:
                shape = Shape(description);
                break;
            case ERROR:

                break;
        }
    }
}

void Descriptor::print() {
    std::cout << "Printing Descriptor:\n";
    shape.print();
    std::cout << "Input_file: " << input_file << "\n";
    std::cout << "Output_file: " << output_file << "\n";
}