#include <iostream>
#include "forward.hpp"

void Running_state::fill_active_nodes_with_input_nodes() {
    Input_Node* input_nodes = nn->get_input_nodes();
    for (int id_node = 0; id_node < nn->get_nb_input_nodes(); id_node++) {
        active_nodes.push(input_nodes[id_node].get_id());
    }
}

Running_state::Running_state(Neural_network* nn) {
    this->nn = nn;
    int nb_nodes = nn->get_nb_nodes();
    nn->set_activation_0();
    fill_active_nodes_with_input_nodes();

    nodes_left_to_activate = new int [nb_nodes];
    for (int id_node = nn->get_nb_input_nodes(); id_node < nb_nodes; id_node++) {
        Output_Node& node = nn->get_output_node(id_node);
        nodes_left_to_activate[id_node] = node.get_nb_parents();
    }
}

Running_state::~Running_state() {
    delete [] nodes_left_to_activate;
}

static void print_queue(std::queue<int> q) {
	while (!q.empty()){
		std::cout << q.front() << " ";
		q.pop();
	}
	std::cout << "\n";
}

void Running_state::print() const {
    std::cout << "Neural Network that we want to run:\n";
    nn->print();

    std::cout << "Current active nodes:\n";
    print_queue(active_nodes);

    std::cout << "Nodes left to activate:\n";
    for (int i = 0; i < nn->get_nb_nodes(); i++) {
        std::cout << nodes_left_to_activate[i] << " ";
    }   
    std::cout << "\n";
}

void Running_state::scan_node(int id_node) {
    nn->compute_node_output(id_node);

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
    }
}

void Running_state::run() {
    while(!active_nodes.empty()) {
        int id_node = active_nodes.front();
        active_nodes.pop();
        scan_node(id_node);
    }
}