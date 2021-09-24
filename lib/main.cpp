#include <iostream>
#include "descriptor.hpp"
#include "information.hpp"
#include "neural_network.hpp"

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    Descriptor desc(argv[1]);
    Information info(desc);
    Neural_network nn(desc, info);
    nn.run();
    nn.print_output();
}