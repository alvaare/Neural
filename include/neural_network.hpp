#pragma once
#include <queue>
#include "descriptor.hpp"
#include "information.hpp"

enum node_type {
    NODE,
    INPUT_NODE,
    OUTPUT_NODE,
    HIDDEN_NODE
};

class Neural_network;

class Node {
    protected:
        int id;
        double output;
        static int nb_nodes;
    public:
        Node();
        virtual void print() const;

        int get_id() const;
        double get_output() const;

        void compute_output() const;
};

class Edge;

class Input_Node : virtual public Node {
    protected:
        Edge* child_edges;
        int nb_childs;
    public: 
        Input_Node();
        ~Input_Node();
        void print() const;

        int get_nb_childs() const;
        Edge* get_childs() const;

        void set_childs(Neural_network&, int, int);
        void set_output(double);

};

class Output_Node : virtual public Node {
    protected:
        double activation;
        double bias;
        int nb_parents;
        int* parents;
        double local_gradient;
    public: 
        Output_Node();
        ~Output_Node();
        void print() const;

        int get_nb_parents() const;
        double get_activation() const;
        double get_local_gradient() const;
        double get_bias() const;

        void set_parents(int, int);
        void set_activation_0();
        void set_gradient_0();
        void increase_activation(double);

        void compute_output();
        void compute_gradient();
};

class Hidden_Node : public Input_Node, public Output_Node {
    private:
        Function activation_function;
    public:
        Hidden_Node();
        Hidden_Node(functions);
        void print() const;

        void compute_output();
        void compute_gradient();
};

class Edge {
    private:
        Output_Node* out_node;
        double weight;
        double output;
        double local_gradient;
    public:
        Edge(Output_Node*);
        Edge();
        void print() const;

        int get_id_out() const;
        double get_weight() const;
        double get_output() const;

        void compute_output(double);
        void compute_gradient(double);
};

class Neural_network {
    private:
        int nb_input_nodes;
        int nb_hidden_nodes;
        int nb_output_nodes;
        Input_Node* input_nodes;
        Hidden_Node* hidden_nodes;
        Output_Node* output_nodes;

    public:
        Neural_network(const Descriptor&, const Information&);
        ~Neural_network();
        void print() const;
        void print_output() const;

        int get_nb_input_nodes() const;
        int get_nb_output_nodes() const;
        int get_nb_nodes() const;
        Input_Node* get_input_nodes() const;
        Input_Node& get_input_node(int) const;
        Output_Node* get_output_nodes() const;
        Output_Node& get_output_node(int) const;
        node_type get_type(int) const;
        Node& get_node(int) const;

        void construct_input_nodes(const Shape&,  int);
        void construct_hidden_nodes(const Shape&, int, int);
        void set_activation_0();
        void set_gradient_0();
        void set_input(double[]);
        void compute_node_output(int);
        void compute_node_gradient(int);

        void run();
        void backpropagate();
};