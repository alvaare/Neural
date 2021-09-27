#pragma once
#include <queue>
#include "neural_network.hpp"

class Running_state {
    private:
        Neural_network* nn;
        std::queue<int> active_nodes;
        int* nodes_left_to_activate;
        bool* visited;

        void fill_active_nodes_with_input_nodes();
        void scan_node(int);
    public:
        Running_state(Neural_network*);
        ~Running_state();
        void print() const;

        void run();



};