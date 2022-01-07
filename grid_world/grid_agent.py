import numpy as np
from numpy import random
import torch
from torch import nn




class NeuralNetwork(nn.Module):
    def __init__(self, sizes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        modules = []
        nb_layers = len(sizes)
        for i in range(nb_layers - 2):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(sizes[nb_layers-2], sizes[nb_layers-1]))
        modules.append(nn.Sigmoid())
        self.linear_relu_stack = nn.Sequential(*modules)

        
    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output



class Species():
    def __init__(self, nb_specimens, nn_size, device, indice, height, width, initial_energy, time_between_matings): #population de la forme n(n-1)/2
        self.color = (int(random.rand()*100) + 155, int(random.rand()*100) + 155, int(random.rand()*100) + 155)
        self.nn_size = nn_size  
        self.nb_specimens = nb_specimens
        self.available_indices = []
        self.specimens = [Specimen(nn_size, indice, device,int(random.rand() * height), int(random.rand() * width), i, initial_energy, time_between_matings) for i in range(self.nb_specimens)]
        self.specimen_list_size = self.nb_specimens
        self.best_specimen = self.specimens[0]    #en theorie, height - 1 ci dessus mais donne des erreurs....
        self.device = device
        self.indice = indice
        self.initial_energy, self.time_between_matings = initial_energy, time_between_matings


    def mate(self, specimenA, specimenB):
        if len(self.available_indices) != 0:
            id = self.available_indices.pop()
        else:
            id = self.specimen_list_size
        
        self.nb_specimens += 1

        specimenA.time_before_mating_possible = specimenB.time_before_mating_possible = self.time_between_matings
        child = Specimen(specimenA.sizes, specimenA.spe_indice, self.device, specimenA.x, specimenA.y, id, self.initial_energy, self.time_between_matings)
        child.energy = (specimenA.energy + specimenB.energy)/2
        specimenA.energy *= .5
        specimenB.energy *= .5


        proba_A = specimenA.energy/(specimenA.energy + specimenB.energy)

        for i in range(len(child.memory)):
            if random.random() < proba_A:
                child.memory[i] = specimenA.memory[i]
            else:
                child.memory[i] = specimenB.memory[i]


        for l in range(len(self.nn_size) -1):
            for w in range(self.nn_size[l]*self.nn_size[l+1]):   #weights loop
                with torch.no_grad():
                    if random.random() < proba_A:
                        v = specimenA.nn.linear_relu_stack[2*l].weight[w//self.nn_size[l]][w%self.nn_size[l]]
                    else:
                        v = specimenB.nn.linear_relu_stack[2*l].weight[w//self.nn_size[l]][w%self.nn_size[l]]

                    child.nn.linear_relu_stack[2*l].weight[w//self.nn_size[l]][w%self.nn_size[l]] = v

            for b in range(self.nn_size[l+1]):   #bias loop
                with torch.no_grad():
                    if random.random() < proba_A:
                        child.nn.linear_relu_stack[2*l].bias[b] = specimenA.nn.linear_relu_stack[2*l].bias[b]
                    else:
                        child.nn.linear_relu_stack[2*l].bias[b] = specimenB.nn.linear_relu_stack[2*l].bias[b]

        child.mutate(2/child.dna_length)

    
        if child.ind_indice == self.specimen_list_size:
            self.specimens.append(child) 
            self.specimen_list_size += 1
        else:
            self.specimens[child.ind_indice] = child 

        return child


    def kill_specimen(self, indice):
        self.specimens[indice] = None
        self.available_indices.append(indice)
        self.nb_specimens -= 1

    def reset(self, nb_specimens):
        self.nb_specimens = nb_specimens
        self.available_indices = []
        self.specimen_list_size = self.nb_specimens


            
class Specimen():
    def __init__(self, sizes, spe_indice, device, x, y, ind_indice, initial_energy, time_between_matings):
        self.x, self.y = x, y
        self.sizes = sizes
        self.spe_indice = spe_indice
        self.ind_indice = ind_indice
        self.nb_layers = len(sizes)
        self.nn = NeuralNetwork(sizes).to(device) 
        self.dna_length = sum([sizes[i+1]*(sizes[i] + 1) for i in range(self.nb_layers - 1)])
        self.energy = initial_energy
        self.time_before_mating_possible = time_between_matings
        self.memory = [0, 0]


    def mutate(self, mut_prob):  #mut_prob  proportionnel à 1/self.dna_length.
        for l in range(self.nb_layers-1):
            for w in range(self.sizes[l]*self.sizes[l+1]):   #weights loop
                if random.random() < mut_prob:
                    r = random.normal(scale=.3)   
                    with torch.no_grad():
                        self.nn.linear_relu_stack[2*l].weight[w//self.sizes[l]][w%self.sizes[l]] += r

            for b in range(self.sizes[l+1]):   #bias loop
                if random.random() < mut_prob:
                    r = random.normal(scale=.3)
                    with torch.no_grad():
                        self.nn.linear_relu_stack[2*l].bias[b] += r 


    
    def take_actions(self, paint, food, player, x, y, energy): #le centre de player est tjrs à 1.0...
        for i,elt in enumerate(player):
            if elt == -1:
                player[i] = 0.
            elif elt == self.spe_indice:
                player[i] = 1.
            else:
                player[i] = -1.
        input = torch.FloatTensor([paint + food + player + [energy, x, y] + self.memory])
        self.nn.eval()
        with torch.no_grad():
            actions = self.nn(input)
            
        return actions[0].tolist()

