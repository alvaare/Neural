#pragma once
#include "descriptor.hpp"
#include "information.hpp"

class Node {
    private:
        int id;
        double output;
    protected:
        static int nb_nodes;
    public:
        Node();
        virtual void print() const;

        int get_id() const;
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

        void set_childs(int, int);
        //void set_output(double);

};

class Output_Node : virtual public Node {
    protected:
        int nb_parents;
        int* parents;
    public: 
        Output_Node();
        ~Output_Node();
        void print() const;

        void set_parents(int, int);
};

class Hidden_Node : public Input_Node, public Output_Node {
    private:
        double bias;
    public:
        Hidden_Node();
        void print() const;
};

class Edge {
    private:
        int id_out;
        double weight;
    public:
        Edge(int);
        Edge();
        void print() const;
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
};