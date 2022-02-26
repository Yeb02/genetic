#include "agent.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;


markers agent::m;  //il faut déclarer les vars statiques hors de la classe ... super !
int agent::in_size, agent::out_size;

#if defined _DEBUG
void agent::test() {
    //cout << << endl;
    cout << node_dna_length << "  " << connection_dna_length << endl;
}
#endif

float sigmoid(float x) {
    return 1/(1+exp(-x));
}

float relu(float x){
    return max(x, (float)0);
}

// un node multiplication peut-être

/*
Un agent est composé de layers, chaque layer a n gènes nodes correspondant aux nodes de cette layer
et c gènes connection qui lient les neurones de cette layers à ceux des précédentes.
*/

agent::agent(int ind, bool has_parents, agent* parentA, agent* parentB) {
    this->agent_indice = ind;
    if (!has_parents){

        this->n_layers = 2;
        this->node_dna_length = in_size + out_size;
        this->node_dna.resize(this->node_dna_length);
        this->connection_dna_length = in_size * out_size;
        this->connection_dna.resize(this->connection_dna_length);
        
        this->nodes_per_layer.resize(this->n_layers);
        this->nodes_per_layer = {in_size, out_size};
        this->connections_per_layer.resize(this->n_layers);
        this->connections_per_layer = {0, in_size*out_size};

        this->node_marking_positions.resize(this->node_dna_length);
        this->connection_marking_positions.resize(this->connection_dna_length);

        node_gene ng;
        ng.is_enabled = true;
        ng.marker = 0;
        ng.in_value = 0.0;
        ng.out_value = 0.0;
        ng.activation = *relu;
        int l = 0;

        for (int i : nodes_per_layer) {
            for (int j = 0; j < i; j++) {
                ng.bias = (float) uniform(rdm);  //random
                node_dna[ng.marker] = ng;
                this->node_marking_positions[ng.marker] = {ng.marker, ng.marker};
                ng.marker++;                    
            }
            l++;
        }

        connection_gene cg;
        cg.is_enabled = true;
        cg.marker = 0;
        l = 0;

        for (int i = 0; i < in_size; i++) {
            for (int j = 0; j < out_size; j++) {
                cg.weight = (float) uniform(rdm);  //random
                cg.input = i;
                cg.output = j + in_size;
                connection_dna[cg.marker] = cg;
                this->connection_marking_positions[cg.marker] = {cg.marker, cg.marker};
                cg.marker++;
            }
            l++;
        }

    } else { // vérifier qu'il s'agit bien de deep copy. parentA est le plus fit.
        this->node_dna_length = parentA->node_dna_length;
        this->node_dna = parentA->node_dna;
        this->connection_dna_length = parentA->connection_dna_length;
        this->connection_dna = parentA->connection_dna;
        
        this->n_layers = parentA->n_layers;
        this->nodes_per_layer = parentA->nodes_per_layer;
        this->connections_per_layer = parentA->connections_per_layer;

        this->node_marking_positions = parentA->node_marking_positions;
        this->connection_marking_positions = parentA->connection_marking_positions;
        

        // on prend la topologie du parent le plus fit (parentA) et là ou les gènes vorrespondent avec parentB on crossover

        //nodes d'abord
        int i_b = 0;    
        for (int i_a=0; i_a < this->node_dna_length; i_a++ ) {
            while (i_b<parentB->node_dna_length && this->node_marking_positions[i_a].marking > parentB->node_marking_positions[i_b].marking) {
                i_b++;
            }

            if (i_b==parentB->node_dna_length) break;
            if (this->node_marking_positions[i_a].marking == parentB->node_marking_positions[i_b].marking) {
                if (uniform(rdm)>.5) {
                    this->node_dna[this->node_marking_positions[i_a].position].bias = parentA->node_dna[parentA->node_marking_positions[i_a].position].bias;
                } else {
                    this->node_dna[this->node_marking_positions[i_a].position].bias = parentB->node_dna[parentB->node_marking_positions[i_b].position].bias;
                }
            }
        }

        // on procède identiquement pour les connections
        i_b = 0;    
        for (int i_a=0; i_a < this->connection_dna_length; i_a++ ) {
            while (i_b<parentB->connection_dna_length && this->connection_marking_positions[i_a].marking > parentB->connection_marking_positions[i_b].marking) {
                i_b++;
            }

            if (i_b==parentB->connection_dna_length) break;
            if (this->connection_marking_positions[i_a].marking == parentB->connection_marking_positions[i_b].marking) {
                if (uniform(rdm)>.5) {
                    this->connection_dna[this->connection_marking_positions[i_a].position].weight = parentA->connection_dna[parentA->connection_marking_positions[i_a].position].weight;
                } else {
                    this->connection_dna[this->connection_marking_positions[i_a].position].weight = parentB->connection_dna[parentB->connection_marking_positions[i_b].position].weight;
                }
            }
        }
    }
}


agent::agent(const agent& parentA) {
    this->agent_indice = -1; //debugging purposes, useless otherwise
    this->node_dna_length = parentA.node_dna_length;
    this->node_dna = parentA.node_dna;
    this->connection_dna_length = parentA.connection_dna_length;
    this->connection_dna = parentA.connection_dna;

    this->n_layers = parentA.n_layers;
    this->nodes_per_layer = parentA.nodes_per_layer;
    this->connections_per_layer = parentA.connections_per_layer;

    this->node_marking_positions = parentA.node_marking_positions;
    this->connection_marking_positions = parentA.connection_marking_positions;
}

vector<float> agent::forward_pass(vector<float> *input) {
    for (node_gene& node : node_dna) {
        node.in_value = 0.0;
    }
    for (int i = 0; i < nodes_per_layer[0]; i++) {
        node_dna[i].out_value = (*input)[i];
    }

    int connection_index = 0;
    int node_index = nodes_per_layer[0];

    for (int i = 1; i < n_layers; i++) {

        for (int j = connection_index; j < connection_index+connections_per_layer[i]; j++){
            if (node_dna[connection_dna[j].input].is_enabled && connection_dna[j].is_enabled) {
                node_dna[connection_dna[j].output].in_value += node_dna[connection_dna[j].input].out_value * connection_dna[j].weight;
            }
        }
        connection_index += connections_per_layer[i];

        for (int j = node_index; j < node_index+nodes_per_layer[i]; j++){
            node_dna[j].out_value = node_dna[j].activation((node_dna[j].in_value+node_dna[j].bias));
        }
        node_index += nodes_per_layer[i];
    }

    vector<float> output;
    output.resize(nodes_per_layer[n_layers-1]);
    for (int i = 0; i < nodes_per_layer[n_layers-1]; i++){
        output[i] = node_dna[node_index - nodes_per_layer[n_layers-1] + i].out_value;
    }
    return output;
}


void agent::mutate(int& n_node_mutations, int& n_connection_mutations, std::vector<new_node_mutation>& new_node_mutations, std::vector<new_connection_mutation>& new_connection_mutations) {

    // mutate weights values
    int n_weight_mutations = (int) 2 + connection_dna_length*uniform(rdm) / 3;
    for (int i = 0; i<n_weight_mutations; i++){
        connection_dna[(int) connection_dna_length*uniform(rdm)].weight += (uniform(rdm)-.5);
    }
    
    // mutate biases values
    int n_bias_mutations = (int) 1 + connection_dna_length * uniform(rdm) / 6;
    for (int i = 0; i<n_bias_mutations; i++){
        node_dna[(int) node_dna_length*uniform(rdm)].bias += (uniform(rdm)-.5);
    }
        
    // add a node in the middle of a connection
    if (uniform(rdm)>.998) {  
        add_node(n_node_mutations, new_node_mutations);
    }

    // add a new connection
    if (uniform(rdm)>.992) {
        add_connection(n_connection_mutations, new_connection_mutations);
    }
}


//int agent::get_node_layer(

void agent::add_node(int& n_node_mutations, std::vector<new_node_mutation>& new_node_mutations){
    // pick random enabled connection, to be changed by recursion,  TODO  !!
    int id; 
    int max_rep = 10;
    do {
        id = floor(connection_dna_length*uniform(rdm));
    } while (!connection_dna[id].is_enabled && max_rep--);

    if (!max_rep) return;

    connection_dna[id].is_enabled = true; //optionnel, TODO


    //determine la position du nouveau noeud et des nouvelles connections
    int l, node_id_at_l, connection_id_at_l = 0;
    int l_in, node_id_at_l_in, connection_id_at_l_in;
    if (nodes_per_layer[0] > connection_dna[id].input) {  // la première layer avec 0 connections est un cas particulier
        l_in = 0;
        node_id_at_l_in = 0;
        connection_id_at_l_in = 0;

        l = 1;
        node_id_at_l = nodes_per_layer[0];
    }  else {
        l = 1;
        node_id_at_l = nodes_per_layer[0];
        while (node_id_at_l + nodes_per_layer[l] <= connection_dna[id].input) {
            node_id_at_l += nodes_per_layer[l];
            connection_id_at_l += connections_per_layer[l];
            l++;
        }
        l_in = l;
        node_id_at_l_in = node_id_at_l;
        connection_id_at_l_in = connection_id_at_l;
    }
   

    while (node_id_at_l + nodes_per_layer[l] <= connection_dna[id].output) {
        node_id_at_l += nodes_per_layer[l];
        connection_id_at_l += connections_per_layer[l];
        l++;
    }
    int l_out = l;

    int node_id_at_l_out = node_id_at_l;
    int connection_id_at_l_out = connection_id_at_l;
    
    // 2 cases:
    int new_node_position, c1_position, c2_position;
    if (l_out-l_in==1){ // if the neuron is between 2 adjacent layers, create new layer right after l_in
        new_node_position = node_id_at_l_in + nodes_per_layer[l_in]; // pareil dans les deux cas en fait

        auto iter1 = nodes_per_layer.begin() + l_in + 1;
        nodes_per_layer.insert(iter1, 1);
        auto iter2 = connections_per_layer.begin() + l_in + 1;
        connections_per_layer.insert(iter2, 1);
        connections_per_layer[l_out+1]++; //+1 car on vient d'insérer la nouvelle layer
        n_layers++;

    }   else { // else add at the beginning of a random layer between the two
        /*cout << "start" << endl;
        cout << l_in << " " << l_out << endl;
        if (l_in == l_out) {
            cout << "wtf" << endl;
        }
        new_node_position = (node_id_at_l_out - node_id_at_l_in - nodes_per_layer[l_in])*uniform(rdm) + node_id_at_l_in + nodes_per_layer[l_in];
        node_id_at_l = node_id_at_l_in;
        l = l_in;
        connection_id_at_l = connection_id_at_l_in;
        for (int i : nodes_per_layer) cout << i << " ";
        cout << endl;
        for (int i : connections_per_layer) cout << i << " ";
        cout << endl;
        cout << new_node_position << " " << endl;
        cout << connection_id_at_l << " " << l << " " << node_id_at_l << " " << endl;
        while (node_id_at_l + nodes_per_layer[l] <= new_node_position) {
            node_id_at_l += nodes_per_layer[l];
            connection_id_at_l += connections_per_layer[l];
            l++;
            cout << connection_id_at_l << " " << l << " " << node_id_at_l << " " << endl;
        }
        new_node_position = node_id_at_l;
        connection_id_at_l_in = connection_id_at_l;
        l_in = l;
        cout << new_node_position << endl;
        nodes_per_layer[l_in]++;
        connections_per_layer[l_in]++;
        connections_per_layer[l_out]++;
        for (int i : nodes_per_layer) cout << i << " ";
        cout << endl;
        for (int i : connections_per_layer) cout << i << " ";
        cout << endl;*/

        new_node_position = node_id_at_l_in + nodes_per_layer[l_in]; 
        nodes_per_layer[l_in+1]++;
        connections_per_layer[l_in+1]++;
        connections_per_layer[l_out]++;
    }

    // faire l'insert dans cet ordre !!!! c1 puis c2, ordre de la propagation
    c1_position = connection_id_at_l_in + connections_per_layer[l_in]; 
    c2_position = connection_id_at_l_out + 1;  // +1 car on insère d'abord c1


    //décalage de l'output des connections pointant sur des nodes après le nouveau 
    for (int j = c1_position; j < connection_dna_length; j++) { 
        connection_dna[j].output++;

        if (connection_dna[j].input >= new_node_position) {
            connection_dna[j].input++;
        }
    }
    
    // décalage de node_marking_position
    for (marking_position& p : node_marking_positions){ // ATTENTION SI p N'EST PAS UNE REFERENCE CA NE CHANGE PAS LA VALEUR (SHALLOW)
        if (p.position >= new_node_position) {
            p.position++;
        }
    }

    // create and insert new node, update everything
    node_gene ng;
    if (uniform(rdm) > .5) {
        ng.activation = *relu;
    } else {
        ng.activation = *sigmoid;
    }
    // first check if this mutation already occured at this generation
    int found_corresponding_mutation = -1;
    for (int i = 0; i < n_node_mutations; i++) {
        if (connection_dna[id].marker == new_node_mutations[i].old_connection_marking && new_node_mutations[i].activation == ng.activation) {  //if has occured
            found_corresponding_mutation = i;
            ng.marker = new_node_mutations[i].new_node_marking;
            break;
        }
    }
    if (found_corresponding_mutation == -1) { //if has not occured
        ng.marker = m.node_marker++;
        new_node_mutations[n_node_mutations] = {connection_dna[id].marker, m.connection_marker, ng.marker, ng.activation}; //TODO source évidente de futurs bugs d'initialiser positionnelement. (2n?2l?2m?)
        n_node_mutations++;
    }
    /*new_node_mutation nm {    on rappelle la structure pour clarifier l'init ci dessus
        int old_connection_marking;
        int new_lower_connection_marking;
        int new_node_marking;
    };*/

    ng.bias = 0.0; 
    ng.in_value = 0.0;
    ng.out_value = 0.0;
    ng.is_enabled=true;
    ng.activation = *relu;
    node_marking_positions.push_back({ng.marker, new_node_position});
    auto iter1 = node_dna.begin() + new_node_position;   
    node_dna.insert(iter1, ng);


    // décalage de connection_marking_position, c2 est à droite de c1 sur le graphe
    for (marking_position& p : connection_marking_positions){
        if (p.position >= c2_position - 1) { // -1 car on décale d'abord à cause de c1 puis à cause de c2, donc le node originellement à c2-1 se retrouve a c2+1
            p.position += 2;                // d'où le +2
        } else if (p.position >= c1_position) {
            p.position++;
        }
    }
    
    // on s'occupe des connections...
    connection_gene c1;
    connection_gene c2;

    if (found_corresponding_mutation == -1) {
        c1.marker = m.connection_marker++;
        c2.marker = m.connection_marker++;
    } else {
        c1.marker = new_node_mutations[found_corresponding_mutation].new_lower_connection_marking;
        c2.marker = c1.marker + 1;
    }

    c1.input = connection_dna[id].input;
    c1.output = new_node_position;
    c1.is_enabled = true;
    c1.weight = 0.0; 
    connection_marking_positions.push_back({c1.marker, c1_position});
    auto iter2 = connection_dna.begin() + c1_position;   
    connection_dna.insert(iter2, c1);

    
    c2.input = new_node_position;
    c2.output = connection_dna[id+1].output;  // +1 dans l'indice puisque c1 a été inséré et pas +1 en dehors car l'output a déjà été actualisé dans la boucle sur connection_dna !!
    c2.is_enabled = true;
    c2.weight = 0.0; 
    connection_marking_positions.push_back({c2.marker, c2_position});
    iter2 = connection_dna.begin() + c2_position;   
    connection_dna.insert(iter2, c2);

    node_dna_length++;
    connection_dna_length += 2;
}

void agent::add_connection(int& n_connection_mutations, std::vector<new_connection_mutation>& new_connection_mutations){
    int out_id, in_id, v1, v2;
    int max_steps_to_find_enabled_nodes;

    for (int i=0; i<5; i++){ //5 tentatives pour trouver deux nodes (enabled) sans connection préexistante
        max_steps_to_find_enabled_nodes = 10; // 10 pour en trouver des enabled
        do {
            v1 = floor(node_dna_length* uniform(rdm));
            v2 = floor(node_dna_length* uniform(rdm));

        } while (!( v1!=v2 && node_dna[v1].is_enabled && node_dna[v2].is_enabled) && max_steps_to_find_enabled_nodes--);
        if (!max_steps_to_find_enabled_nodes) continue;
        in_id = min(v1, v2);
        out_id = max(v1, v2);

        // déterminer les indices necessaire
        // déterminer les indices necessaire
        //la premiere layer est un cas particulier puisque 0 connections, on la saute
        int l=1, node_id_at_l=nodes_per_layer[0], connection_id_at_l=0;
        while (node_id_at_l + nodes_per_layer[l] <= out_id) {
            connection_id_at_l += connections_per_layer[l];
            node_id_at_l += nodes_per_layer[l];
            l++;
        }

        //les deux nodes sont-ils sur la même layer ?
        if (node_id_at_l<=in_id || out_id<=1) continue;

        // les deux nodes sont-ils déjà connectés ?
        bool connection_already_exists = false;
        for (int j = connection_id_at_l; j<connection_id_at_l+connections_per_layer[l]; j++) {
            if (connection_dna[j].input == in_id && connection_dna[j].output == out_id) {
                connection_already_exists = true;
                break;
            }
        }
        if (connection_already_exists) continue;
        

        // update everything
        int new_connection_position = connection_id_at_l + connections_per_layer[l];
        connection_gene cg;

        // but first check if this mutation already occured at this gen
        int found_corresponding_mutation = -1;
        for (int i = 0; i < n_connection_mutations; i++) {
            if (node_dna[in_id].marker == new_connection_mutations[i].input_node_marking && node_dna[out_id].marker == new_connection_mutations[i].output_node_marking) {  //if has occured
                found_corresponding_mutation = i;
                cg.marker = new_connection_mutations[i].new_connection_marking;
                break;
            }
        }
        if (found_corresponding_mutation == -1) { //if has not occured
            cg.marker = m.connection_marker++;
            new_connection_mutations[n_connection_mutations] = { node_dna[in_id].marker, node_dna[out_id].marker, cg.marker};
            n_connection_mutations++;
        }
        /*struct new_connection_mutation {   RAPPEL
            int input_node_marking;
            int output_node_marking;
            int new_connection_marking;
        };*/

        cg.input = in_id;
        cg.output = out_id;
        cg.is_enabled = true;
        cg.weight = 0.0; 
        auto iter = connection_dna.begin() + new_connection_position;   
        connection_dna.insert(iter, cg);
        connections_per_layer[l]++;
        connection_dna_length++;
        for (marking_position& p : connection_marking_positions){
            if (p.position >= new_connection_position) {
                p.position++;
            }
        }
        connection_marking_positions.push_back({cg.marker, new_connection_position});

        break;
    }
}


void agent::compute_compatibility(agent* b, agent::compatibility_characteristics * pcc){ 
    int n_common_nodes=0, n_common_connections=0;
    // nodes d'abord:
    int i_b = 0;    
    for (int i_a=0; i_a < node_dna_length; i_a++ ) {
        while (i_b<b->node_dna_length && node_marking_positions[i_a].marking > b->node_marking_positions[i_b].marking) {
            i_b++;
            pcc->n_disjoint_node_genes++;
        }
        if (i_b==b->node_dna_length) {
            pcc->n_excess_node_genes = node_dna_length - i_a;
            break;
        }
        if (node_marking_positions[i_a].marking == b->node_marking_positions[i_b].marking) {
            // average weight since bias can be seen as a weight from a constant neuron 
            pcc->average_weight_difference += abs(node_dna[node_marking_positions[i_a].position].bias - b->node_dna[b->node_marking_positions[i_b].position].bias);
            i_b++;
            n_common_nodes++;
            n_common_connections++; //parce que le bias est compté comme un weight
        } else {
            pcc->n_disjoint_node_genes++;
        }
    }
    if (i_b != b->node_dna_length) { 
        pcc->n_excess_node_genes = b->node_dna_length - i_b;
    }


    //connections:
    i_b=0;
    for (int i_a=0; i_a < connection_dna_length; i_a++ ) {
        while (i_b<b->connection_dna_length && connection_marking_positions[i_a].marking > b->connection_marking_positions[i_b].marking) {
            i_b++;
            pcc->n_disjoint_connection_genes++;
        }
        if (i_b==b->connection_dna_length) {
            pcc->n_excess_connection_genes = connection_dna_length - i_a;
            break;
        }

        if (connection_marking_positions[i_a].marking == b->connection_marking_positions[i_b].marking) {
            pcc->average_weight_difference += abs(connection_dna[connection_marking_positions[i_a].position].weight - b->connection_dna[b->connection_marking_positions[i_b].position].weight);
            n_common_connections++;
            i_b++;
        } else {
            pcc->n_disjoint_connection_genes++;
        }
    }
    if (i_b != b->connection_dna_length) {
        pcc->n_excess_connection_genes = b->connection_dna_length - i_b;
    }
    if (n_common_connections > 0) {
        pcc->average_weight_difference /= n_common_connections;
    }
}

void agent::draw(sf::RenderWindow* window, int x_offset, int y_offset) {

    //store the drawn nodes'positions
    vector<float> node_x;
    vector<float> node_y;
    node_x.resize(node_dna_length);
    node_y.resize(node_dna_length);
    
    // reuse the same shapes over and over again.
    sf::CircleShape base_node(10.f);
    base_node.setFillColor(sf::Color::White);

    sf::Vertex base_connection[] =
    {
        sf::Vertex(sf::Vector2f(0.f, 0.f)),
        sf::Vertex(sf::Vector2f(0.f, 0.f))
    };


    //drawing, the more visible a connection the higher the weight. Red is positive, blue is negative. 
    int connection_color;
    int connection_id = 0, node_id = 0;
    for (int layer = 0; layer < n_layers; layer++) {
        //nodes first
        for (int n_id = node_id; n_id < node_id + nodes_per_layer[layer]; n_id++) {
            node_y[n_id] = 300 + 40*(n_id-node_id - (float)nodes_per_layer[layer]/2) + y_offset;
            node_x[n_id] = (layer + 2) * 50 + x_offset;
            base_node.setPosition(node_x[n_id], node_y[n_id]);

            if (node_dna[n_id].activation != *relu) {   //YELLOW IF SIGMOID, GREEN FOR RELU
                base_node.setFillColor(sf::Color::Yellow);
                cout << "SIGMOID" << endl;
            } else {
                base_node.setFillColor(sf::Color::Green);
            }

            window->draw(base_node);
        }
        node_id += nodes_per_layer[layer];

        //connections 
        for (int c_id = connection_id; c_id < connection_id + connections_per_layer[layer]; c_id++) {
            if (!connection_dna[c_id].is_enabled) {
                connection_color = min(100 + (int)abs(connection_dna[c_id].weight) * 100, 255);
                base_connection[0].color = sf::Color(0, connection_color, 0);
                base_connection[1].color = sf::Color(0, connection_color, 0);
            } else if (connection_dna[c_id].weight > 0) {
                connection_color = min(100 + (int) connection_dna[c_id].weight * 100, 255);
                base_connection[0].color = sf::Color(connection_color, 0, 0);
                base_connection[1].color = sf::Color(connection_color, 0, 0);
            } else {
                connection_color = min(100 + (int) -connection_dna[c_id].weight * 100, 255);
                base_connection[0].color = sf::Color(0, 0, connection_color);
                base_connection[1].color = sf::Color(0, 0, connection_color);
            }
            base_connection[0].position = { 10 + node_x[connection_dna[c_id].input],  10 + node_y[connection_dna[c_id].input]};
            base_connection[1].position = { 10 + node_x[connection_dna[c_id].output], 10 + node_y[connection_dna[c_id].output]};
            window->draw(base_connection, 2, sf::Lines);
        }
        connection_id += connections_per_layer[layer];
    }
}

int agent::get_node_dna_length() {
    return node_dna_length;
}
int agent::get_connection_dna_length() {
    return connection_dna_length;
}


float agent::evaluate_fitness(float data[100][2]) {
    long double s = 0;
    vector<float> input = { 0.0, 0.0 };
    for (int i = 0; i < 100; i++) {
        input[0] = data[i][0];
        input[1] = data[i][1];

        s += pow(abs(forward_pass(&input)[0] - (data[i][0] * data[i][1])), 2);
        //s += pow(pow(forward_pass(&input)[0] - (data[i][0]*data[i][1]), 2) + .3, -1);    // = 1/(diff^2 + .1)
    }
    return -sqrt(s)/ 10.0; 
}

