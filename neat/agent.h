#pragma once

#include <vector>
#include <random>
#include <SFML/Graphics.hpp>



extern std::random_device rdm;
extern std::uniform_real_distribution<> uniform;

struct node_gene {
    int marker;
    float bias;
    bool is_enabled;
    float in_value;
    float out_value;
    float (*activation) (float);
};

struct connection_gene {
    int input;
    int output;
    int marker;
    float weight;
    bool is_enabled;
};

struct marking_position {
    int marking;
    int position;
};

struct markers {
    int node_marker;
    int connection_marker;
};

struct new_node_mutation { // TODO fonctions d'activation
    int old_connection_marking;
    int new_lower_connection_marking;
    int new_node_marking;
    float (*activation) (float);
};

struct new_connection_mutation {
    int input_node_marking;
    int output_node_marking;
    int new_connection_marking;
};


class agent{
    public:
        static int in_size, out_size;
        static markers m;
        int agent_indice, species_id;
        
    private:
        int n_layers;
        int connection_dna_length, node_dna_length;
        std::vector<node_gene> node_dna;
        std::vector<connection_gene> connection_dna;
        
        std::vector<int> nodes_per_layer;
        std::vector<int> connections_per_layer;

        std::vector<marking_position> node_marking_positions;
        std::vector<marking_position> connection_marking_positions;

    public:
        struct compatibility_characteristics{
            int n_disjoint_node_genes;
            int n_disjoint_connection_genes;
            int n_excess_node_genes;
            int n_excess_connection_genes;
            float average_weight_difference;
        };

        //agent();

#if defined _DEBUG
        void test();
#endif

        agent(int ind, bool has_parents, agent* parentA=nullptr, agent* parentB=nullptr);
        agent(const agent& agent);
        std::vector<float> forward_pass(std::vector<float> *input);
        void mutate(int& n_node_mutations, int& n_connection_mutations, std::vector<new_node_mutation>& new_node_mutations, std::vector<new_connection_mutation>& new_connection_mutations);
        float evaluate_fitness(float data[100][2]);
        void compute_compatibility(agent* b, compatibility_characteristics* pcc);
        void draw(sf::RenderWindow* window, int x_offset, int y_offset);
        int get_node_dna_length(); 
        int get_connection_dna_length();

    private:
        void add_node(int& n_node_mutations, std::vector<new_node_mutation>& new_node_mutations);
        void add_connection(int& n_connection_mutations, std::vector<new_connection_mutation>& new_connection_mutations);
        
};
