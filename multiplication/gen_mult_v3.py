#task : compute multiplication of ints.
import argparse
import numpy as np
from numpy import random
import torch
from torch import nn


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

        self.function_to_approximate = lambda x,y : x*y
        


    def evolve(self, nb_generations, nb_calculs_for_fitness):
        for i in range(nb_generations):
            best_fitness_at_this_generation = self.time_step(nb_calculs_for_fitness)
            if i%10 == 0:
                print(f"Mean quadratic error of best specimen at generation {i}:  {best_fitness_at_this_generation}")
                nb_calculs_test = 100
                x = torch.Tensor([[10*random.rand(), 10*random.rand()] for _ in range(nb_calculs_test)])
                data = [x.to(device), torch.Tensor([[self.function_to_approximate(x[j][0],x[j][1])] for j in range(nb_calculs_test)]).to(device)]
                print('all times best is at  ', self.best_specimen.evaluate_fitness(data)/nb_calculs_test)
            if i >= 200 and (i-200+1)%100 == 0:
                r = (random.rand()-.5) * 10
                print(f"CHANGED GOAL, f(x,y) = x*y + {r}")
                self.function_to_approximate =  lambda x,y : x*y + r
                

        nb_calculs_test = 100
        x = torch.Tensor([[10*random.rand(), 10*random.rand()] for _ in range(nb_calculs_test)])
        data = [x.to(device), torch.Tensor([[self.function_to_approximate(x[j][0],x[j][1])] for j in range(nb_calculs_test)]).to(device)]
        return self.best_specimen, self.best_specimen.evaluate_fitness(data)/nb_calculs_test


    def time_step(self, nb_calculs):
        # x = torch.Tensor([[10*random.rand(), 10*random.rand()] for _ in range(nb_calculs)])
        x = torch.Tensor([[u+random.rand()-.5, v+random.rand()-.5] for u in range(1, 11) for v in range(1, 11)])
        data = [x.to(device), torch.Tensor([[self.function_to_approximate(x[j][0],x[j][1])] for j in range(nb_calculs)]).to(device)]

        for i in range(self.nb_specimens):
            self.fitness_scores[i] = self.specimens[i].evaluate_fitness(data)/nb_calculs

        elite_id = [k for k in range(self.nb_specimens)]
        elite_id.sort(key=lambda i : self.fitness_scores[i])
        elite_id = elite_id[:self.elite_nb] #indice dans specimens des elite_nb plus fits
        elite = [self.specimens[id] for id in elite_id]
        elite_fitness = [self.fitness_scores[id] for id in elite_id]

        if elite_fitness[0] < self.best_specimen.evaluate_fitness(data)/nb_calculs:
            self.best_specimen = elite[0]
            self.best_specimen_fitness = elite_fitness[0]
        # else:
        #     elite[-1] = self.best_specimen 
        #     elite_fitness[-1] = self.best_specimen_fitness


        # print(elite[0].mutation_buffers[7:10])


        self.specimens = self.mate_elite(elite, elite_fitness)
        for specimen in self.specimens:
            specimen.mutate()

        return elite_fitness[0]

    def mate_elite(self, elite, elite_fitness):
        next_gen = []
        for i in range(self.elite_nb):
            for j in range(i+1, self.elite_nb):
                next_gen.append(self.mate(elite[i], elite[j], elite_fitness[i], elite_fitness[j]))
        return next_gen
        


    def mate(self, specimenA, specimenB, fitA, fitB):
        child = Specimen(self.nn_size)
        fromA =  fitB/(fitA+fitB)
        fromA =  1/(1+np.exp(-((fromA-.5) * 30)))
        with torch.no_grad():
            id = 0

            for l in range(child.nb_layers-1):
                for w in range(child.sizes[l]*child.sizes[l+1]): 
                    if random.random() < fromA:
                        child.mutation_buffers[id] = specimenA.mutation_buffers[id]
                        child.nn.linear_relu_stack[2*l].weight[w//child.sizes[l]][w%child.sizes[l]] = \
                            specimenA.nn.linear_relu_stack[2*l].weight[w//child.sizes[l]][w%child.sizes[l]]

                    else:
                        child.mutation_buffers[id] = specimenB.mutation_buffers[id]
                        child.nn.linear_relu_stack[2*l].weight[w//child.sizes[l]][w%child.sizes[l]] = \
                            specimenB.nn.linear_relu_stack[2*l].weight[w//child.sizes[l]][w%child.sizes[l]]

                    id += 1

                for b in range(child.sizes[l+1]): 
                    if random.random() < fromA:
                        child.mutation_buffers[id] = specimenA.mutation_buffers[id]
                        child.nn.linear_relu_stack[2*l].bias[b] = specimenA.nn.linear_relu_stack[2*l].bias[b]

                    else:
                        child.mutation_buffers[id] = specimenB.mutation_buffers[id]
                        child.nn.linear_relu_stack[2*l].bias[b] = specimenB.nn.linear_relu_stack[2*l].bias[b]
                    
                    id += 1

        return child


            
class Specimen():
    def __init__(self, sizes):  
        self.sizes = sizes
        self.nb_layers = len(sizes)
        self.nn = NeuralNetwork(sizes).to(device) 

        self.dna_length = sum([sizes[i+1]*(sizes[i] + 1) for i in range(self.nb_layers - 1)])

        self.mut_prob = 5/self.dna_length #~.05
        self.buffer_size = 3
        self.mutation_buffers = [[1]*self.buffer_size]*self.dna_length



    def mutate(self):  
        with torch.no_grad():
            id = 0
            buffer = [1]*self.buffer_size
            for l in range(self.nb_layers-1):
                for w in range(self.sizes[l]*self.sizes[l+1]): 
                    if random.random() < self.mut_prob*buffer[0]:
                        self.nn.linear_relu_stack[2*l].weight[w//self.sizes[l]][w%self.sizes[l]] += random.random()-.5
                        buffer = buffer[1:]
                        buffer.append(1)
                        self.mutation_buffers[id] = [u*((1.05)**(1-2*(random.random()>.5))) for u in self.mutation_buffers[id]]
                        buffer = [a*b for a, b in zip(self.mutation_buffers[id], buffer)]
                    id += 1
                    # if id == 30 and random.random()<.001:
                        # print(buffer, self.mut_prob, "WTF IS GOOOINNNG OOONNNNNNNNNNNNNNNNN")


                for b in range(self.sizes[l+1]): 
                    if random.random() < self.mut_prob*buffer[0]:
                        self.nn.linear_relu_stack[2*l].bias[b] += random.random()-.5
                        buffer = buffer[1:]
                        buffer.append(1)
                        self.mutation_buffers[id] = [u*(1.05**(1-2*(random.random()>.5))) for u in self.mutation_buffers[id]]
                        buffer = [a*b for a, b in zip(self.mutation_buffers[id], buffer)]
                    id += 1


    def evaluate_fitness(self, data): 
        self.nn.eval()
        mean_dev = 0
        with torch.no_grad():
            pred = self.nn(data[0])
            mean_dev += ((pred - data[1])**2).type(torch.float).sum().item()   
        return mean_dev




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_gen', type=int, 
                    help='Number of generations', default=3000)

    parser.add_argument('--no_evo', dest='no_evolution', action='store_const', const=True,
                        default=False,
                        help='disables evolution process')

    parser.add_argument('--test', dest='test', action='store_const', const=True,
                        default=False,
                        help='lets you test the results !')

    args = parser.parse_args()


    
    if not args.no_evolution:
        Boulier = Species(15, [2, 20, 1])
        ElCalculator, accuracy = Boulier.evolve(args.nb_gen, 100)
        print(f"Error of best specimen = {accuracy}")
        acc_str = str(int(100*accuracy))
        name = "ElCalculator_" + acc_str[0] + "." + acc_str[1:] + ".pth"
        torch.save(ElCalculator.nn.state_dict(), name)

    if args.test:
        model = NeuralNetwork([2, 15, 1])
        model.load_state_dict(torch.load("ElCalculator.pth"))

        x = input()
        while x != "q":
            a, b = x.split(" ")
            print(f"{float(a)} x {float(b)} = {model(torch.Tensor([[float(b), float(a)]]))[0][0]}.")
            x = input()
