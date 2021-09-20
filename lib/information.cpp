#include <fstream>
#include <iostream>
#include "information.hpp"

int get_dimmension(std::string file) {
    std::string line;
    std::ifstream myfile (file);
    int dimmension = 0;
    getline(myfile, line);
    for (char c : line) {
        if (c==' ') {
            dimmension++;
        }
    }    
    return dimmension+1;
}

int get_nb_lines(std::string file) {
    std::string line;
    std::ifstream myfile (file);
    int nb_lines = 0;
    while (getline(myfile, line)) {
        nb_lines++;
    }
    return nb_lines;
}

Dimmensions::Dimmensions(Descriptor& desc) {
    const std::string input_file = desc.get_input_file();
    const std::string output_file = desc.get_output_file();

    dimmension_input = get_dimmension(input_file);
    dimmension_output = get_dimmension(output_file);
    nb_trainings = get_nb_lines(input_file);
}

int Dimmensions::get_nb_trainings() const {
    return nb_trainings;
}

int Dimmensions::get_dimmension_input() const {
    return dimmension_input;
}

int Dimmensions::get_dimmension_output() const {
    return dimmension_output;
}

Information::Information(Descriptor& desc) :
    dimmensions(desc),
    inputs(dimmensions.get_nb_trainings(), dimmensions.get_dimmension_input(), desc.get_input_file()),
    outputs(dimmensions.get_nb_trainings(), dimmensions.get_dimmension_output(), desc.get_output_file())
{}

void Information::print() {
    std::cout << "Printing Information:\n";
    std::cout << "Number of trainings: " << dimmensions.get_nb_trainings() << "\n";
    std::cout << "Dimmension input: " << dimmensions.get_dimmension_input() << "\n";
    std::cout << "Dimmension output: " << dimmensions.get_dimmension_output() << "\n";
    std::cout << "Inputs:\n";
    inputs.print();
    std::cout << "Outputs:\n";
    outputs.print();
}