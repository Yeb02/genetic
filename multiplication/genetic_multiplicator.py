#task : compute multiplication of ints.
import argparse
import numpy as np
from numpy import random
import torch
from torch import nn

POPULATION = (8-1)*8/2
SPECIES = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

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
        self.linear_relu_stack = nn.Sequential(*modules)

        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Species():
    def __init__(self, population_base, nn_size): #population de la forme n(n-1)/2
        self.nn_size = nn_size
        self.elite_nb = population_base
        self.nb_specimens = int(population_base * (population_base - 1)/2)
        self.specimens = [Specimen(nn_size) for _ in range(self.nb_specimens)]
        self.fitness_scores = [0]*self.nb_specimens
        self.best_specimen = self.specimens[0]
        self.best_specimen_fitness = 1e3


    def evolve(self, nb_generations, nb_calculs_for_fitness):
        for i in range(nb_generations):
            best_fitness_at_this_generation = self.time_step(nb_calculs_for_fitness)
            if i%10 == 0:
                print(f"Mean quadratic error of best at gen {i}:  ", format(best_fitness_at_this_generation, ".1f"), "   VARIANCE :  ", format(np.var(np.array(self.fitness_scores)),".1f"))
                nb_calculs_test = 100
                x = torch.Tensor([[10*random.rand(), 10*random.rand()] for j in range(nb_calculs_test)])
                data = [x.to(device), torch.Tensor([[x[j][0]*x[j][1]] for j in range(nb_calculs_test)]).to(device)]
                print(f'all times best is at  ', format(self.best_specimen.evaluate_fitness(data)/nb_calculs_test, ".1f"))

        nb_calculs_test = 100
        x = torch.Tensor([[10*random.rand(), 10*random.rand()] for j in range(nb_calculs_test)])
        data = [x.to(device), torch.Tensor([[x[j][0]*x[j][1]] for j in range(nb_calculs_test)]).to(device)]
        return self.best_specimen, self.best_specimen.evaluate_fitness(data)/nb_calculs_test


    def time_step(self, nb_calculs):
        x = torch.Tensor([[10*random.rand(), 10*random.rand()] for j in range(nb_calculs)])
        data = [x.to(device), torch.Tensor([[x[j][0]*x[j][1]] for j in range(nb_calculs)]).to(device)]

        for i in range(self.nb_specimens):
            self.fitness_scores[i] = self.specimens[i].evaluate_fitness(data)/nb_calculs

        elite_id = [k for k in range(self.nb_specimens)]
        elite_id.sort(key=lambda i : self.fitness_scores[i])
        elite_id = elite_id[:self.elite_nb] #indice dans specimens des elite_nb plus fits
        elite = [self.specimens[id] for id in elite_id]
        elite_fitness = [self.fitness_scores[id] for id in elite_id]

        if elite_fitness[0] < self.best_specimen_fitness:
            self.best_specimen = elite[0]
            self.best_specimen_fitness = elite_fitness[0]
        # else:
        #     elite[-1] = self.best_specimen 
        #     elite_fitness[-1] = self.best_specimen_fitness

        self.specimens = self.mate_elite(elite, elite_fitness)
        for specimen in self.specimens:
            specimen.mutate(2/specimen.dna_length)

        return elite_fitness[0]

    def mate_elite(self, elite, elite_fitness):
        next_gen = []
        for i in range(self.elite_nb):
            for j in range(i+1, self.elite_nb):
                next_gen.append(self.mate(elite[i], elite[j], elite_fitness[i], elite_fitness[j]))
        return next_gen
        


    def mate(self, specimenA, specimenB, fitA, fitB):
        child = Specimen(self.nn_size)
        indice = 0
        layer = 0
        is_weight = True

        probA = fitB/(fitA + fitB)       #fonction basique, la deuxieme ligne utilise une sigmoide. C'est fitB car plus bas est fit meilleur est l'individu
        probA = 1/(1+np.exp(-((probA-.5) * 50)))
         
        for i in range(specimenA.dna_length):
            # si rdm < alpha/(alpha+beta), on prend A. alpha = fitA/gene_strength_A, idem beta et B
            fromA = random.random() < probA
            if not is_weight and indice >= self.nn_size[layer+1]:
                indice = 0
                is_weight = True
                layer += 1
             
            if is_weight and indice < self.nn_size[layer+1]*self.nn_size[layer]: #c'est un weight
                with torch.no_grad():
                    if fromA:
                        child.nn.linear_relu_stack[2*layer].weight[indice//self.nn_size[layer]][indice%self.nn_size[layer]] = \
                            specimenA.nn.linear_relu_stack[2*layer].weight[indice//self.nn_size[layer]][indice%self.nn_size[layer]]
                    else:
                        child.nn.linear_relu_stack[2*layer].weight[indice//self.nn_size[layer]][indice%self.nn_size[layer]] = \
                            specimenB.nn.linear_relu_stack[2*layer].weight[indice//self.nn_size[layer]][indice%self.nn_size[layer]]

            elif is_weight:  #à la transition des weights aux bias
                is_weight = False
                indice -= self.nn_size[layer+1]*self.nn_size[layer]
                with torch.no_grad():
                    if fromA:
                        child.nn.linear_relu_stack[2*layer].bias[indice] = specimenA.nn.linear_relu_stack[2*layer].bias[indice]
                    else:
                        child.nn.linear_relu_stack[2*layer].bias[indice] = specimenB.nn.linear_relu_stack[2*layer].bias[indice]

            else:  # c'est un bias
                with torch.no_grad():
                    if fromA:
                        child.nn.linear_relu_stack[2*layer].bias[indice] = specimenA.nn.linear_relu_stack[2*layer].bias[indice]
                    else:
                        child.nn.linear_relu_stack[2*layer].bias[indice] = specimenB.nn.linear_relu_stack[2*layer].bias[indice]

            indice += 1

        return child


            
class Specimen():
    def __init__(self, sizes):  
        self.sizes = sizes
        self.nb_layers = len(sizes)
        self.nn = NeuralNetwork(sizes).to(device) 
        self.dna_length = sum([sizes[i+1]*(sizes[i] + 1) for i in range(self.nb_layers - 1)])


    def mutate(self, mut_prob):  #mut_prob  proportionnel à 1/self.dna_length. Avec 2/dna length:

        for l in range(self.nb_layers-1):
            for w in range(self.sizes[l]*self.sizes[l+1]):   #weights loop
                self.update_weights(mut_prob, l, w)  

            for b in range(self.sizes[l+1]):   #bias loop
                self.update_bias(mut_prob, l, b)  


    def update_weights(self, mut_prob, l, w):
         
        if random.random() < mut_prob:
            r = random.normal(scale=.3)   
            with torch.no_grad():
                self.nn.linear_relu_stack[2*l].weight[w//self.sizes[l]][w%self.sizes[l]] += r


    def update_bias(self, mut_prob, l, b):
        
        if random.random() < mut_prob:
            r = random.normal(scale=.3)
            with torch.no_grad():
                self.nn.linear_relu_stack[2*l].bias[b] += r

        


    def evaluate_fitness(self, data): #THE LOWER THE BETTER
        self.nn.eval()
        mean_dev = 0
        with torch.no_grad():
            pred = self.nn(data[0])
            mean_dev += ((pred-data[1])**2).type(torch.float).sum().item()   
        return mean_dev




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_gen', type=int,
                    help='Number of generations', default=100)

    parser.add_argument('--no_evo', dest='no_evolution', action='store_const',  const=True,
                        default=False,
                        help='disables evolution process')

    parser.add_argument('--test', dest='test', action='store_const',  const=True,
                        default=False,
                        help='lets you test the results !')

    args = parser.parse_args()


    
    if not args.no_evolution:
        Boulier = Species(10, [2, 20, 1])
        ElCalculator, accuracy = Boulier.evolve(args.nb_gen, 50)
        print(f"Error of best specimen = {accuracy}")
        acc_str = str(int(100*accuracy))
        name = "ElCalculator_" + acc_str[0] + "." + acc_str[1:] + ".pth"
        torch.save(ElCalculator.nn.state_dict(), name)

    if args.test:
        model = NeuralNetwork([2, 20, 1])
        model.load_state_dict(torch.load("ElCalculator.pth"))

        x = input()
        while x != "q":
            a, b = x.split(" ")
            print(f"{float(a)} x {float(b)} = {model(torch.Tensor([[float(b), float(a)]]))[0][0]}.")
            x = input()
