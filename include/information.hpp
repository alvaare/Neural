#pragma once
#include "descriptor.hpp"
#include "matrix.hpp"

class Dimmensions {
    private:
        int nb_trainings;
        int dimmension_input;
        int dimmension_output;
    public:
        Dimmensions(Descriptor&);
        int get_nb_trainings() const;
        int get_dimmension_input() const;
        int get_dimmension_output() const;
};

class Information {
    private:
        Dimmensions dimmensions;
        Matrix inputs;
        Matrix outputs;

    public:
        Information(Descriptor&);

        void print();
};