#pragma once

#include "agent.h"

#include <iostream>

extern std::random_device rdm;
extern std::uniform_real_distribution<> uniform;

int binary_search(std::vector<float>& proba, int len_proba, float value);
        
class population{
    public: 
        float C1, C2, C3, C4, C5, COMPATIBILITY_THRESHOLD;
        int N_SPECIMEN, IN_SIZE, OUT_SIZE, N_SPECIES;
        int steps;
        int fittest_individual;

    private:
        std::vector<agent*> specimens;
        std::vector<float> fitnesses;
        std::vector<float> probabilities;
        std::vector<float> compatibilities;

        int actual_species_number;
        std::vector<agent*> species_core_specimens;
        std::vector<int> specimen_per_species;

        std::vector<new_node_mutation> new_node_mutations;
        int n_node_mutations;
        std::vector<new_connection_mutation> new_connection_mutations;
        int n_connection_mutations;

    public:
        // c0: avg_b_diff, c1: avg_w_diff, c2: n_disjoint_n, c3: n_disjoint_c, c4: n_excess_n, c5: n_excess_c, le tout divisé par les dna_length respectifs
        population(float C1=1.0, float C2=1.0, float C3=.4, float TRESHOLD=1,
                   int N_SPECIMEN=100, int IN_SIZE=2, int OUT_SIZE=1, int N_SPECIES=10);
        bool run_one_evolution_step();
        void draw(sf::RenderWindow* window, sf::Font& font);

#if defined _DEBUG
        void test();
#endif

    private:
        bool evaluate_fitnesses();
        void speciate();
        void mate();
        void mutate();
};
