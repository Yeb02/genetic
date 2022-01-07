#pragma once

#include "agent.h"
#include <vector>
#include <cmath>
#include <random>

using namespace std;


typedef struct gene {
    int input;
    int output;
    int marker;
    float parameter; // either weight or bias, whether it is a connection or a node gene.
    bool is_enabled;
    bool is_node_gene;   // tells whether it is a connection or a node gene.
};

float sigmoid(float x) {
    return 1/(1+exp(-x));
}

class agent {
    public:

        static int in_size, out_size;
        int indice;

        int dna_length;
        vector<gene> dna;

    private:
        int n_layers;
        
        vector<int> nodes_per_layer;
        vector<int> connections_per_layer;

    public: 
        agent(int ind, bool has_parents) {
            this->indice = ind;
            
            if (!has_parents){
                this->dna_length = (in_size+1)*(out_size+1)-1;
                this->dna.resize(this->dna_length);
                this->n_layers = 2;
                this->nodes_per_layer = {in_size, out_size};
                this->connections_per_layer = {in_size*out_size};

                int dna_enumerator = 0;
                gene g;
                g.is_enabled = true;
                g.is_node_gene = true;
                g.marker = 1;

                for (int j = 0; j < in_size; j++) {
                    g.parameter = rand()-.5;
                    g.input = j;
                    g.output = j;
                    dna.push_back(g);
                    dna_enumerator++;
                }
                
                g.is_node_gene = false;
                for (int j = 0; j < in_size; j++) {
                    for (int k = 0; k < in_size; k++) {
                        g.parameter = rand()-.5;
                        g.input = j;
                        g.output = j;
                        dna.push_back(g);
                        dna_enumerator++;
                    }
                }

                g.is_node_gene = true;
                for (int j = 0; j < out_size; j++) {
                    g.parameter = rand()-.5;
                    g.input = j;
                    g.output = j;
                    dna.push_back(g);
                    dna_enumerator++;
                }

            } else {
                
            }
        }

    
    public: 
        vector<float> forward_pass(vector<float> input) {
            vector<float> current_output;
            vector<float> previous_output = input;

            int dna_enumerator = 0;

            for (int i = 0; i < n_layers; i++) {

                current_output.resize(nodes_per_layer[i+1]);
                for (int j = 0; j < nodes_per_layer[i+1]; j++) {
                    current_output[j] = 0.0;
                }
                

                for (int j = 0; j < connections_per_layer[i]; j++) {
                    if (dna[dna_enumerator].is_enabled) {
                        current_output[dna[dna_enumerator].output] += dna[dna_enumerator].parameter * previous_output[dna[dna_enumerator].input]; 
                    }
                    dna_enumerator++;
                }

                for (int j = 0; j < nodes_per_layer[i+1]; j++) {
                    if (dna[dna_enumerator].is_enabled) {
                        current_output[j] = sigmoid(current_output[j]+dna[dna_enumerator].parameter);
                    }
                    dna_enumerator++;
                }
                
                previous_output = current_output;
            }

            return current_output;
        }
    


};
