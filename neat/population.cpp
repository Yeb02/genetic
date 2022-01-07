#pragma once

#include <vector>
#include "population.h"
#include "agent.h"

using namespace std;

class population {
    public: 
        float C0, C1, C2, TRESHOLD;
        int N_SPECIMEN, IN_SIZE, OUT_SIZE;

        population(int N_SPECIMEN=100, float C0=1.0, float C1=1.0, float C2=.4, float TRESHOLD=3.0, int IN_SIZE=2, int OUT_SIZE=1) {
            this->C0 = C0;
            this->C1 = C1;
            this->C2 = C2;
            this->TRESHOLD = TRESHOLD;
            this->N_SPECIMEN = N_SPECIMEN;
            this->IN_SIZE = IN_SIZE;
            this->OUT_SIZE = OUT_SIZE;
            this->specimens.resize(this->N_SPECIMEN);
            
            agent::in_size = IN_SIZE;
            agent::out_size = OUT_SIZE;

            for (int i=0; i<this->N_SPECIMEN; i++){

                this->specimens.push_back(agent(i, false));
            }

        }

    private:
        vector<agent> specimens;   
        
};