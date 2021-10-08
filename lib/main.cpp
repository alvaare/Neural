#include <iostream>
#include "descriptor.hpp"
#include "information.hpp"
#include "neural_network.hpp"

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    Descriptor desc(argv[1]);
    Information info(desc);
    Neural_network nn(desc, info);
    double inputs[2] = {0.0, 1.0};
    nn.set_input(inputs);
    nn.run();
    nn.print_output();
    nn.backpropagate();
}