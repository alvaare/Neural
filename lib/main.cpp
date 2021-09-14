#include <iostream>
#include "descriptor.hpp"
#include "neural_network.hpp"

int main(int argc, char* argv[]) {
    Descriptor desc(argv[1]);
    desc.print();
}