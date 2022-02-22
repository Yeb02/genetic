#include "population.h"

using namespace std;

int binary_search(vector<float>& proba, int len_proba, float value){ // returns parent.
    int inf = 0;
    int sup = len_proba - 1;

    if (proba[0] > value) { 
        return 0;
    }

    int mid;
    int max_iter = 100;
    while (max_iter--) {
        if (sup - inf <= 1) return proba[inf] > value ? inf : sup;  // sinon n'évalue que inf indéfiniment qd inf + 1 = sup

        mid = (sup + inf)/2;
        if (proba[mid-1] <= value && value < proba[mid]) {
            return mid;
        } else if (proba[mid] < value) {
            inf = mid;
        } else {
            sup = mid;
        }
    }
    cout << "failure" << endl;
    return inf; // max_iter was reached, thus we return the arbitrary value inf. A security.
}

#if defined _DEBUG
void population::test() {
    /*cout << N_SPECIMEN << endl;
    for (int i = 0; i < N_SPECIMEN; i++) {
        cout << specimens[i]->agent_indice << endl;
    }*/
}
#endif

population::population(float C0, float C1, float C2, float C3, float C4, float C5, float THRESHOLD, 
                       int N_SPECIMEN, int IN_SIZE, int OUT_SIZE, int N_SPECIES)
    : C0(C0), C1(C1), C2(C2), C3(C3), C4(C4), C5(C5), COMPATIBILITY_THRESHOLD(THRESHOLD),
      N_SPECIMEN(N_SPECIMEN), IN_SIZE(IN_SIZE), OUT_SIZE(OUT_SIZE), N_SPECIES(N_SPECIES)
    
    { 

    this->specimens.resize(this->N_SPECIMEN);
    this->fitnesses.resize(this->N_SPECIMEN);
    this->probabilities.resize(this->N_SPECIMEN);
    this->compatibilities.resize(this->N_SPECIMEN); // wont need resizing even if new species are created
    this->steps = 0;
    this->fittest_individual = 0;

    agent::in_size = IN_SIZE;
    agent::out_size = OUT_SIZE;
    agent::m = { IN_SIZE + OUT_SIZE, IN_SIZE * OUT_SIZE };


    for (int i=0; i<this->N_SPECIMEN; i++){
        this->specimens[i] = new agent(i, false);
    }

    // species initialisation
    actual_species_number = N_SPECIES;
    species_core_specimens.resize(N_SPECIMEN);
    specimen_per_species.resize(N_SPECIMEN);
    for (int j=0; j<N_SPECIES; j++) { 
        species_core_specimens[j] = new agent(-1, false);  // no copying of the first generation because of destructions at the end of speciation
    }

    // to keep track of mutation at each generation
    new_node_mutations.resize(N_SPECIMEN); //N_SPECIMEN est overkill mais ne prenons pas de risques
    new_connection_mutations.resize(N_SPECIMEN);
    n_node_mutations = 0;   // pas necessaire, reset à 0 à chaque génération
    n_connection_mutations = 0;
}  

void population::mate() {
    vector<agent*> temp_population;
    temp_population.resize(N_SPECIMEN);
    int i_a, i_b;

    for (int i=0; i<N_SPECIMEN; i++){
        i_a = binary_search(probabilities, N_SPECIMEN, uniform(rdm));

        do {
            i_b = binary_search(probabilities, N_SPECIMEN, uniform(rdm));
        } while (i_b == i_a);

        if (fitnesses[i_a]>fitnesses[i_b]){
            temp_population[i] = new agent(i, true, specimens[i_a], specimens[i_b]);
        } else {
            temp_population[i] = new agent(i, true, specimens[i_b], specimens[i_a]);
        }
    }

    for (agent* a : specimens) delete a;
    specimens = temp_population;
}

void population::mutate() {
    n_node_mutations = 0;   
    n_connection_mutations = 0;

    for (agent* a : specimens){
        a->mutate(n_node_mutations, n_connection_mutations, new_node_mutations, new_connection_mutations); //N_SPECIMEN est overkill mais ne prenons pas de risques
    }
}

void population::speciate() {
    // these must be resized to the full size each time because of the erasures.
    species_core_specimens.resize(N_SPECIMEN);

    agent::compatibility_characteristics* pcc = new agent::compatibility_characteristics{};
    int most_compatible_species;
    vector<bool> has_core_for_next_gen;
    has_core_for_next_gen.resize(actual_species_number);

    for (int i = 0; i < actual_species_number; i++) {
        specimen_per_species[i] = 0;
        has_core_for_next_gen[i] = false;
    }

    int new_species = 0;

    for (int i=0; i<N_SPECIMEN; i++) {
        most_compatible_species = 0;

        for (int j=0; j<actual_species_number; j++) {

            pcc->n_disjoint_node_genes = 0;
            pcc->n_disjoint_connection_genes = 0;
            pcc->n_excess_node_genes = 0;
            pcc->n_excess_connection_genes = 0;
            pcc->average_bias_difference = 0.0;
            pcc->average_weight_difference = 0.0;

            specimens[i]->compute_compatibility(species_core_specimens[j], pcc);

            compatibilities[j] = C0*pcc->average_bias_difference  + (C2 * pcc->n_disjoint_node_genes + C4 * pcc->n_excess_node_genes)/ specimens[i]->get_node_dna_length() +
                                 C1*pcc->average_weight_difference + (C3 * pcc->n_disjoint_connection_genes + C5 * pcc->n_excess_connection_genes) / specimens[i]->get_connection_dna_length();
                                  
            if (compatibilities[j] < compatibilities[most_compatible_species]) {
                most_compatible_species = j;
            }
        }
           
        if (compatibilities[most_compatible_species] < COMPATIBILITY_THRESHOLD) { // if close enough, goes to the nearest species
            specimens[i]->species_id = most_compatible_species;
            specimen_per_species[most_compatible_species]++;
            if (!has_core_for_next_gen[most_compatible_species]) {
                //cout << most_compatible_species << endl;
                specimen_per_species[most_compatible_species] = 1;
                delete species_core_specimens[most_compatible_species];
                species_core_specimens[most_compatible_species] = new agent(*specimens[i]);
                has_core_for_next_gen[most_compatible_species] = true;
            }
        } else {  // otherwise create new species. If the compatibility threshold is too low (or there is a bug...)there will be more species than there are individuals
            delete species_core_specimens[actual_species_number + new_species];  // le bug de mutation se manifeste ici
            species_core_specimens[actual_species_number + new_species] = new agent(*specimens[i]);
            specimen_per_species[actual_species_number + new_species] = 1;
            specimens[i]->species_id = actual_species_number + new_species;
            new_species++;
        }
    }

    // keep species that have more than 0 individuals among those from last gen
    int alive_species=0, erasure_counter=0;
    //cout << specimen_per_species[0] << " " << specimen_per_species[1] << " " << specimen_per_species[2] << " " << specimen_per_species[3] << " " << endl;
    for (int i = 0; i < actual_species_number; i++) {
        if (specimen_per_species[i] == 0) {
            delete species_core_specimens[i - erasure_counter];
            species_core_specimens.erase(species_core_specimens.begin() + i - erasure_counter);
            //specimen_per_species NE DOIT PAS CHANGER !!!!
            erasure_counter++;
        } else {
            alive_species++;
        }
    }

    actual_species_number = new_species + alive_species;
}


void population::run_one_evolution_step() {

    // main functions
    speciate();
    evaluate_fitnesses();
    mate();
    mutate();
    steps++;

    if (steps % 100 == 99) {
        cout << "done with step    " << steps+1 << endl;

        vector<float> input = { 7.0, 6.0 };
        cout << "7*6 = " << specimens[fittest_individual]->forward_pass(&input)[0] << endl;
        if (fittest_individual > 0) {
            cout << "most dominant specimen has likelihood   " << probabilities[fittest_individual]- probabilities[fittest_individual-1] << "   of being chosen as a parent" << endl;
        } else {
            cout << "most dominant specimen has likelihood   " << probabilities[fittest_individual] << "   of being chosen as a parent" << endl;
        }
        
#if defined _DEBUG
#endif 
    }
}

void population::evaluate_fitnesses() {  //TODO la fonction ici et dans le calcul du fitness individuel doit dépendre de la tache

    //multiplication data creation
    float data[100][2];
    for (int i=0; i<100; i++){
        data[i][0] = { i/10 + (float) uniform(rdm) -.5f};
        data[i][1] = { i%10 + (float) uniform(rdm) -.5f};
    }
    
    // individual fitnesses
    float max_fitness = 0.0;
    float max_proba = 0.0;
    long double probability_normalization_factor = 0.0;
    for (int i = 0; i < N_SPECIMEN; i++) {

        fitnesses[i] = specimens[i]->evaluate_fitness(data)*pow(specimen_per_species[specimens[i]->species_id], .1);
        if (fitnesses[i] > max_fitness) {
            max_fitness = fitnesses[i];
            //fittest_individual = i;
        }

        // TOUT CE QUI SUIT EST MODULABLE EST DEPEND DE LA TACHE A ACCOMPLIR

        //probabilities[i] = pow(4, fitnesses[i]);
        ////probabilities[i] = fitnesses[i];

        //probability_normalization_factor += probabilities[i];
    }
    for (int i = 0; i < N_SPECIMEN; i++) {
        probabilities[i] = max_fitness - fitnesses[i] + 100/fitnesses[i];
        probability_normalization_factor += probabilities[i];
        if (probabilities[i] < max_proba) {
            max_proba = probabilities[i];
            fittest_individual = i;
        }
    }
    if (probability_normalization_factor == 0) {  // un specime a un meilleur fitness que les autres suite à une mutation, mais les autres ont tous le même, extremement proche du pire
        cout << "???" << endl;                    // il
    }
    //cout << probability_normalization_factor << endl;
    //cout <<  "max_fitness:    " << max_fitness << endl;

    //create the probability-sum array used for picking 2 parents in the mating process  
    probabilities[0] = probabilities[0]/ probability_normalization_factor;
    for (int i = 1; i < N_SPECIMEN; i++) {
        probabilities[i] = probabilities[i-1] + probabilities[i]/ probability_normalization_factor;
    }

}


void population::draw(sf::RenderWindow* window) {
    // Some monitoring. Could use a dedicated function.
   /* if (fittest_individual >= 2) {
        cout << "probabilities:   " << probabilities[fittest_individual]- probabilities[fittest_individual-1] << "  " << probabilities[fittest_individual-1] - probabilities[fittest_individual-2] << endl;
        cout << "fitnesses:   " << fitnesses[fittest_individual] << "  " << fitnesses[fittest_individual-1] << endl;
    } else {
        cout << "probabilities:   " << probabilities[fittest_individual] << "  " << probabilities[fittest_individual+1] - probabilities[fittest_individual] << endl;
        cout << "fitnesses:   " << fitnesses[fittest_individual] << "  " << fitnesses[fittest_individual+1] << endl;
    }*/

    //draw
    specimens[fittest_individual]->draw(window);
}
