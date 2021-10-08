#include <queue>
#include "neural_network.hpp"


class Reverse_State {
    private: 
        Neural_network* nn;
        std::queue<int> active_nodes;
        int* nodes_left_to_activate;
        Function cost_function;

        void fill_active_nodes_with_output_nodes();
        void scan_node(int);

    public:
        Reverse_State(Neural_network*);
        ~Reverse_State();

        void print() const;

        void run();

};