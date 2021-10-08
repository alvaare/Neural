#include <iostream>
#include "backpropagation.hpp"

void Reverse_State::fill_active_nodes_with_output_nodes() {
    Output_Node* output_nodes = nn->get_output_nodes();
    for (int id_node = 0; id_node < nn->get_nb_output_nodes(); id_node++) {
        active_nodes.push(output_nodes[id_node].get_id());
    }
}

Reverse_State::Reverse_State(Neural_network* nn) {
    this->nn = nn;
    int nb_nodes = nn->get_nb_nodes();
    nn->set_gradient_0();
    fill_active_nodes_with_output_nodes();

    nodes_left_to_activate = new int [nb_nodes];
    for (int id_node = 0; id_node < nb_nodes-(nn->get_nb_output_nodes()); id_node++) {
        Input_Node& node = nn->get_input_node(id_node);
        node.print();
        nodes_left_to_activate[id_node] = node.get_nb_childs();
    }
}

Reverse_State::~Reverse_State() {
    delete [] nodes_left_to_activate;
}

void print_queue(std::queue<int> q) {
	while (!q.empty()){
		std::cout << q.front() << " ";
		q.pop();
	}
	std::cout << "\n";
}

void Reverse_State::print() const {
    std::cout << "Neural Network that we want to backpropagate:\n";
    nn->print();

    std::cout << "Current active nodes:\n";
    print_queue(active_nodes);

    std::cout << "Nodes left to activate:\n";
    for (int i = 0; i < nn->get_nb_nodes()-nn->get_nb_output_nodes(); i++) {
        std::cout << nodes_left_to_activate[i] << " ";
    }   
    std::cout << "\n";
}

void Reverse_State::scan_node(int id_node) {
    nn->compute_node_gradient(id_node);
    /*nn->compute_node_output(id_node);

    if (nn->get_type(id_node)==OUTPUT_NODE) 
        return;

    Input_Node& node = dynamic_cast<Input_Node&>(nn->get_node(id_node));
    int nb_childs = node.get_nb_childs();
    Edge* childs = node.get_childs();
    for (int id_edge = 0; id_edge < nb_childs; id_edge++) {
        Edge& edge = childs[id_edge];
        edge.compute_output(node.get_output());

        int id_node_of_child = edge.get_id_out();
        nodes_left_to_activate[id_node_of_child]--;
        if (nodes_left_to_activate[id_node_of_child] == 0) {
            active_nodes.push(id_node_of_child);
        }

        Output_Node& child = dynamic_cast<Output_Node&>(nn->get_node(id_node_of_child));
        child.increase_activation(edge.get_output());
    }*/
}

void Reverse_State::run() {
    print();
    while(!active_nodes.empty()) {
        int id_node = active_nodes.front();
        active_nodes.pop();
        scan_node(id_node);
    }
}