#include <iostream>
#include "neural_network.hpp"
#include "forward.hpp"
#include "backpropagation.hpp"

static double my_rand() {
    return ((double) rand() / (RAND_MAX));
}

int Node::nb_nodes = 0;

Node::Node() {
    output = 0;
    id = nb_nodes;
    nb_nodes++;
}

void Node::print() const {
    std::cout << "id: " << id << "\n";
    std::cout << "output: " << output << "\n";
}

int Node::get_id() const {
    return id;
}

double Node::get_output() const {
    return output;
}

void Node::compute_output() const {
    std::cout << "Computing output of a non-specified node, unknown behaviour.\n";
}

Input_Node::Input_Node() : Node() {
    child_edges = NULL;
    nb_childs = 0;
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

int Input_Node::get_nb_childs() const {
    return nb_childs;
}

Edge* Input_Node::get_childs() const {
    return child_edges;
}

void Input_Node::set_childs(Neural_network& nn, int nb_childs, int start) {
    this->nb_childs = nb_childs;
    child_edges = new Edge[nb_childs];
    for (int id_child = 0; id_child < nb_childs; id_child++) {
        child_edges[id_child] = Edge(dynamic_cast<Output_Node*>(&nn.get_node(start+id_child)));
    }
} 

void Input_Node::set_output(double o) {
    this->output = o;
}

Output_Node::Output_Node() : Node() {
    parents = NULL;
    bias = my_rand();
    activation = 0;
    local_gradient = 0;
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

int Output_Node::get_nb_parents() const {
    return nb_parents;
}

double Output_Node::get_bias() const {
    return bias;
}

double Output_Node::get_activation() const {
    return activation;
}

double Output_Node::get_local_gradient() const {
    return local_gradient;
}

void Output_Node::set_parents(int nb_parents, int start) {
    this->nb_parents = nb_parents;
    parents = new int[nb_parents];
    for (int id_parent = 0; id_parent < nb_parents; id_parent++) {
        parents[id_parent] = start+id_parent;
    }
}

void Output_Node::set_activation_0() {
    activation = 0;
}

void Output_Node::set_gradient_0() {
    local_gradient = 0;
}

void Output_Node::increase_activation(double incr) {
    activation += incr;
}

void Output_Node::compute_output() {
    output = activation;
}

void Output_Node::compute_gradient() {
    
}

Hidden_Node::Hidden_Node() : Input_Node(), Output_Node() {
    activation_function = Function(RELU);
}

Hidden_Node::Hidden_Node(functions f) : Hidden_Node() {
    activation_function = Function(RELU);
}

void Hidden_Node::print() const {
    Node::print();
    std::cout << "This is a hidden node.\n";
    std::cout << "bias: " << bias << "\n";
    std::cout << "activation: " << activation << "\n";
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

void Hidden_Node::compute_output() {
    output = activation_function.f(activation+bias);
}

void Hidden_Node::compute_gradient() {

}

Edge::Edge(Output_Node* n) {
    out_node = n;
    weight = my_rand();
    local_gradient = 0;
    output = 0;
}

Edge::Edge() = default;

void Edge::print() const {
    std::cout << "id: " << out_node->get_id() << " weight: " << weight << " output: " << output << "\n";
}

int Edge::get_id_out() const {
    return out_node->get_id();
}

double Edge::get_weight() const {
    return weight;
}

double Edge::get_output() const {
    return output;
}

void Edge::compute_output(double in) {
    output = in*weight;
}

void Edge::compute_gradient(double activation) {
    local_gradient = activation * out_node->get_local_gradient();
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

void Neural_network::construct_input_nodes(const Shape& shape, int nb_input_nodes) {
    int nb_childs = shape.get_size_layer(0);
    for (int id_input_node = 0; id_input_node < nb_input_nodes; id_input_node++) {
        input_nodes[id_input_node].set_childs(*this, nb_childs, nb_input_nodes);
    }
}

void Neural_network::construct_hidden_nodes(const Shape& shape, int nb_input_nodes, int nb_output_nodes) {
    int nb_childs, nb_parents, nb_visited_nodes;
    nb_visited_nodes = nb_input_nodes;
    for (int id_layer = 0; id_layer < shape.get_depth(); id_layer++) {
        nb_childs = get_nb_childs(id_layer, shape, nb_output_nodes);
        nb_parents = get_nb_parents(id_layer, shape, nb_input_nodes);
        int last_id = nb_visited_nodes + shape.get_size_layer(id_layer);
        int first_id = nb_visited_nodes - nb_parents;
        for (int id_hidden_node = nb_visited_nodes-nb_input_nodes; id_hidden_node < last_id-nb_input_nodes; id_hidden_node++) {
            hidden_nodes[id_hidden_node].set_childs(*this, nb_childs, last_id);
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

    construct_input_nodes(shape, nb_input_nodes);
    construct_hidden_nodes(shape, nb_input_nodes, nb_output_nodes);
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

int Neural_network::get_nb_input_nodes() const {
    return nb_input_nodes;
}

int Neural_network::get_nb_output_nodes() const {
    return nb_output_nodes;
}

int Neural_network::get_nb_nodes() const {
    return nb_hidden_nodes + nb_input_nodes + nb_output_nodes;
}

Input_Node* Neural_network::get_input_nodes() const {
    return input_nodes;
}

Input_Node& Neural_network::get_input_node(int id) const {
    if (id < nb_input_nodes)
        return input_nodes[id];
    return hidden_nodes[id-nb_input_nodes];
}

Output_Node* Neural_network::get_output_nodes() const {
    return output_nodes;
}

Output_Node& Neural_network::get_output_node(int id) const {
    if (id < nb_input_nodes + nb_hidden_nodes) {
        return hidden_nodes[id-nb_input_nodes];
    }
    else    
        return output_nodes[id-nb_input_nodes-nb_hidden_nodes];
}

node_type Neural_network::get_type(int id_node) const {
    if (id_node < nb_input_nodes) {
        return INPUT_NODE;
    }
    if (id_node < nb_input_nodes + nb_hidden_nodes) {
        return HIDDEN_NODE;
    }
    if (id_node < get_nb_nodes()) {
        return OUTPUT_NODE;
    }
    return NODE;
}

Node& Neural_network::get_node(int id_node) const {
    switch (get_type(id_node)) {
    case INPUT_NODE:
        return input_nodes[id_node];
    case HIDDEN_NODE:
        return hidden_nodes[id_node-nb_input_nodes];
    case OUTPUT_NODE:
        return output_nodes[id_node-nb_input_nodes-nb_hidden_nodes];
    default:
        std::cout << "Id_node non existant\n";
        return input_nodes[0];
    }
}

void Neural_network::set_activation_0() {
    for (int id_hidden_node = 0; id_hidden_node < nb_hidden_nodes; id_hidden_node++) {
        hidden_nodes[id_hidden_node].set_activation_0();
    }
    for (int id_output_node = 0; id_output_node < nb_output_nodes; id_output_node++) {
        output_nodes[id_output_node].set_activation_0();
    }
}

void Neural_network::set_gradient_0() {
    for (int id_hidden_node = 0; id_hidden_node < nb_hidden_nodes; id_hidden_node++) {
        hidden_nodes[id_hidden_node].set_gradient_0();
    }
    for (int id_output_node = 0; id_output_node < nb_output_nodes; id_output_node++) {
        output_nodes[id_output_node].set_gradient_0();
    }
}

void Neural_network::set_input(double inputs[]) {
    for (int id_input = 0; id_input < nb_input_nodes; id_input++) {
        input_nodes[id_input].set_output(inputs[id_input]);
    }
}

void Neural_network::compute_node_output(int id_node) {
    Node* node = &get_node(id_node);
    Hidden_Node* h_node;
    Output_Node* o_node;
    switch (get_type(id_node)) {
        case HIDDEN_NODE:
            h_node = dynamic_cast<Hidden_Node*>(node);
            h_node->compute_output();
            break;
        case OUTPUT_NODE:
            o_node = dynamic_cast<Output_Node*>(node);
            o_node->compute_output();
            break;
        default:
            break;
    }
}

void Neural_network::compute_node_gradient(int id_node) {
    Node* node = &get_node(id_node);
    Hidden_Node* h_node;
    Output_Node* o_node;
    switch (get_type(id_node)) {
        case HIDDEN_NODE:
            h_node = dynamic_cast<Hidden_Node*>(node);
            h_node->compute_gradient();
            break;
        case OUTPUT_NODE:
            o_node = dynamic_cast<Output_Node*>(node);
            o_node->compute_gradient();
            break;
        default:
            break;
    }
}

void Neural_network::print_output() const {
    for (int id_output = 0; id_output < nb_output_nodes; id_output++) {
        std::cout << output_nodes[id_output].get_output() << " ";
    }
    std::cout << "\n";
}

void Neural_network::run() {
    Running_state rs(this);
    rs.run();
}

void Neural_network::backpropagate() {
    Reverse_State rs(this);
    rs.run();
}