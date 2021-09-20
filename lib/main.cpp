#include <iostream>
#include "descriptor.hpp"
#include "information.hpp"
#include "neural_network.hpp"

int main(int argc, char* argv[]) {
    Descriptor desc(argv[1]);
    Information info(desc);
    info.print();
}