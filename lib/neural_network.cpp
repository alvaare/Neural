#include <iostream>
#include "neural_network.hpp"

static double my_rand() {
    return ((double) rand() / (RAND_MAX));
}

int Node::nb_nodes = 0;

Node::Node() {
    id = nb_nodes;
    nb_nodes++;
}

void Node::print() const {
    std::cout << "id: " << id << "\n";
}

int Node::get_id() const {
    return id;
}

Input_Node::Input_Node() : Node() {
    child_edges = NULL;
}

Input_Node::~Input_Node() {
    delete [] child_edges;
}

void Input_Node::print() const {
    std::cout << "This is an input node.\n";
    Node::print();
    std::cout << "Here are the childs of the node:\n";
    for (int id_child = 0; id_child < nb_childs; id_child++) {
        child_edges[id_child].print();
    }
}

void Input_Node::set_childs(int nb_childs, int start) {
    this->nb_childs = nb_childs;
    child_edges = new Edge[nb_childs];
    for (int id_child = 0; id_child < nb_childs; id_child++) {
        child_edges[id_child] = Edge(start+id_child);
    }
} 

Hidden_Node::Hidden_Node() : Input_Node(), Output_Node() {
    bias = my_rand();
}

void Hidden_Node::print() const {
    Node::print();
    std::cout << "This is a hidden node.\n";
    std::cout << "bias: " << bias << "\n";
    std::cout << "Here are the childs of the node:\n";
    for (int id_child = 0; id_child < nb_childs; id_child++) {
        child_edges[id_child].print();
    }
    std::cout << "Here are the parents of the node:\n";
    for (int id_parent = 0; id_parent < nb_parents; id_parent++) {
        std::cout << parents[id_parent] << " ";
    }
    std::cout << "\n";
}

Output_Node::Output_Node() : Node() {
    parents = NULL;
} 

Output_Node::~Output_Node() {
    delete [] parents;
}

void Output_Node::print() const {
    std::cout << "This is an output node.\n";
    Node::print();
    std::cout << "Here are the parents of the node:\n";
    for (int id_parent = 0; id_parent < nb_parents; id_parent++) {
        std::cout << parents[id_parent] << " ";
    }
    std::cout << "\n";
}

void Output_Node::set_parents(int nb_parents, int start) {
    this->nb_parents = nb_parents;
    parents = new int[nb_parents];
    for (int id_parent = 0; id_parent < nb_parents; id_parent++) {
        parents[id_parent] = start+id_parent;
    }
}

Edge::Edge(int n) {
    id_out = n;
    weight = my_rand();
}

Edge::Edge() = default;

void Edge::print() const {
    std::cout << "id: " << id_out << " weight: " << weight << "\n";
}

static int compute_nb_hidden_nodes(const Shape& shape) {
    int res = 0;

    for (int id_layer = 0; id_layer < shape.get_depth(); id_layer++) {
        res += shape.get_size_layer(id_layer);
    }

    return res;             
}

static int get_nb_childs(int id_layer, const Shape& shape, int nb_output_nodes) {
    if (id_layer == shape.get_depth() - 1) {
        return nb_output_nodes;
    }
    return shape.get_size_layer(id_layer+1);
}

static int get_nb_parents(int id_layer, const Shape& shape, int nb_input_nodes) {
    if (id_layer == 0) {
        return nb_input_nodes;
    }
    return shape.get_size_layer(id_layer-1);
}

static void construct_input_nodes(const Shape& shape, int nb_input_nodes, Input_Node* input_nodes) {
    int nb_childs = shape.get_size_layer(0);
    for (int id_input_node = 0; id_input_node < nb_input_nodes; id_input_node++) {
        input_nodes[id_input_node].set_childs(nb_childs, nb_input_nodes);
    }

    int nb_parents = shape.get_size_layer(shape.get_depth()-1);
    for (int id_output_node = 0; id_output_node < nb_output_nodes; id_output_node++) {
        output_nodes[id_output_node].set_parents(nb_parents, nb_hidden_nodes - nb_parents + nb_input_nodes);
}

static void construct_hidden_nodes(const Shape& shape, int nb_input_nodes, int nb_output_nodes, Hidden_Node* hidden_nodes) {
    int nb_childs, nb_parents, nb_visited_nodes;
    nb_visited_nodes = nb_input_nodes;
    for (int id_layer = 0; id_layer < shape.get_depth(); id_layer++) {
        nb_childs = get_nb_childs(id_layer, shape, nb_output_nodes);
        nb_parents = get_nb_parents(id_layer, shape, nb_input_nodes);
        int last_id = nb_visited_nodes + shape.get_size_layer(id_layer);
        int first_id = nb_visited_nodes - nb_parents;
        for (int id_hidden_node = nb_visited_nodes-nb_input_nodes; id_hidden_node < last_id-nb_input_nodes; id_hidden_node++) {
            hidden_nodes[id_hidden_node].set_childs(nb_childs, last_id);
            hidden_nodes[id_hidden_node].set_parents(nb_parents, first_id);
        }
        nb_visited_nodes = last_id;
    }
}

static void construct_output_nodes(const Shape& shape, int nb_output_nodes, int nb_visited_nodes, Output_Node* output_nodes) {
    int nb_parents = shape.get_size_layer(shape.get_depth()-1);
    for (int id_output_node = 0; id_output_node < nb_output_nodes; id_output_node++) {
        output_nodes[id_output_node].set_parents(nb_parents, nb_visited_nodes - nb_parents);
    }
}

Neural_network::Neural_network(const Descriptor& desc, const Information& info) {
    const Shape& shape = desc.get_shape();
    const Dimmensions& dimmensions = info.get_dimmensions();
    
    nb_input_nodes = dimmensions.get_dimmension_input();
    nb_hidden_nodes = compute_nb_hidden_nodes(shape);
    nb_output_nodes = dimmensions.get_dimmension_output();
    
    input_nodes = new Input_Node[nb_input_nodes];
    hidden_nodes = new Hidden_Node[nb_hidden_nodes];
    output_nodes = new Output_Node[nb_output_nodes];

    construct_input_nodes(shape, nb_input_nodes, input_nodes);
    construct_hidden_nodes(shape, nb_input_nodes, nb_output_nodes, hidden_nodes);
    construct_output_nodes(shape, nb_output_nodes, nb_input_nodes+nb_hidden_nodes, output_nodes);
}

Neural_network::~Neural_network() {
    delete [] input_nodes;
    delete [] hidden_nodes;
    delete [] output_nodes;
}

void Neural_network::print() const {
    for (int id = 0; id < nb_input_nodes; id++) {
        input_nodes[id].print();
    }
    for (int id = 0; id < nb_hidden_nodes; id++) {
        hidden_nodes[id].print();
    }
    for (int id = 0; id < nb_output_nodes; id++) {
        output_nodes[id].print();
    }
}