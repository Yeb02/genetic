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

        self.out_size = sizes[-1]
        modules = []
        nb_layers = len(sizes)
        for i in range(nb_layers - 2):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            # modules.append(nn.ReLU())
            modules.append(nn.Sigmoid())
        modules.append(nn.Linear(sizes[nb_layers-2], sizes[nb_layers-1]))
        modules.append(nn.Sigmoid())

        self.linear_relu_stack = nn.Sequential(*modules)

        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) - torch.tensor([.5]*self.out_size)
        return logits



def test(model, loss_fn, nb_batches, samples_per_batch, in_size):  #ne s'applique que pour un réseau identité
    model.eval()
    test_loss, mean_dev = 0, 0
    with torch.no_grad():
        for _ in range(nb_batches):

            x = torch.Tensor([[random.randn() for _ in range(in_size)] for j in range(samples_per_batch)])
            # y = x.clone().detach()
            y = torch.Tensor([[0 for _ in range(in_size)] for j in range(samples_per_batch)])
            x, y = x.to(device), y.to(device)

            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            mean_dev += (abs(pred-y)).type(torch.float).sum().item()
            
    test_loss /= nb_batches
    mean_dev /= nb_batches*samples_per_batch
    return test_loss, mean_dev

def train(model, loss_fn, optimizer, nb_batches, samples_per_batch, in_size):  #ne s'applique que pour un réseau identité
    model.train()
    for _ in range(nb_batches):
        x = torch.tensor([[random.randn() for _ in range(in_size)] for j in range(samples_per_batch)])
        # y = x.clone().detach()
        y = torch.Tensor([[0 for _ in range(in_size)] for j in range(samples_per_batch)])
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


class Species():


    def __init__(self, population_base, action_nn_size, mutation_nn_size): #mutation_nn_size est pour un réseau identité 
        self.action_nn_size = action_nn_size
        self.mutation_nn_size = mutation_nn_size


        mutation_nn = NeuralNetwork(mutation_nn_size).to(device) 
        try:
            mutation_nn.load_state_dict(torch.load("pretrained_mutation_net.pth"))
        except:
            pass

        epochs = 501
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(mutation_nn.parameters(), lr=5e-3)
        for t in range(epochs):
            train(mutation_nn, loss_fn, optimizer, 100, 10, mutation_nn_size[0])
            if t%100==0:
                print(f"Epoch {t}")
                test_loss, mean_dev = test(mutation_nn, loss_fn, 1, 100, mutation_nn_size[0])
                print(f"test loss = {test_loss}, mean_dev = {mean_dev}")
        torch.save(mutation_nn.state_dict(), "pretrained_mutation_net.pth")

        self.nb_specimens = int(population_base * (population_base - 1)/2)   #population de la forme n(n-1)/2
        self.fitness_scores = [0]*self.nb_specimens
        self.specimens = [Specimen(action_nn_size, mutation_nn_size, mutation_nn) for _ in range(self.nb_specimens)]
        self.elite_nb = population_base
        self.best_specimen = self.specimens[0]
        self.best_specimen_fitness = 1e3

        

    def evolve(self, nb_generations, nb_calculs_for_fitness):
        for i in range(nb_generations):
            best_fitness_at_this_generation = self.time_step(nb_calculs_for_fitness)
            if i%1 == 0:
                print(f"Mean quadratic error of best specimen at generation {i}:  {best_fitness_at_this_generation}")
                nb_calculs_test = 100
                x = torch.Tensor([[10*random.rand(), 10*random.rand()] for j in range(nb_calculs_test)])
                data = [x.to(device), torch.Tensor([[x[j][0]*x[j][1]] for j in range(nb_calculs_test)]).to(device)]
                print('all times best is at  ', self.best_specimen.evaluate_fitness(data)/nb_calculs_test)

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

        # else:     #TODO trouver une meilleur variation de cette idée
        #     elite[-1] = self.best_specimen 
        #     elite_fitness[-1] = self.best_specimen_fitness

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
        child = Specimen(self.action_nn_size, self.mutation_nn_size, None)

        for i in range(specimenA.dna_length):
            # si rdm < alpha/(alpha+beta), on prend A. alpha = fitA/gene_strength_A, idem beta et B
            fromA = random.random() < fitA/(fitA + fitB)

            if fromA:
                child.dna[i] = specimenA.dna[i]
            else:
                child.dna[i] = specimenB.dna[i]
            
            child.update_networks()

        return child


            
class Specimen():
    def __init__(self, action_nn_size, mutation_nn_size, mutation_nn):  
        self.action_nn_size = action_nn_size
        self.nb_action_layers = len(action_nn_size)
        self.action_nn = NeuralNetwork(action_nn_size).to(device)

        self.mutation_nn_size = mutation_nn_size
        self.nb_mutation_layers = len(mutation_nn_size)

        if mutation_nn is not None:           #Appelé pour les individues de premiere génération
            self.mutation_nn = mutation_nn     #TODO overfit le nn sur son adn
        else:
            self.mutation_nn = NeuralNetwork(self.mutation_nn_size)

        self.action_dna_length = sum([action_nn_size[i+1]*(action_nn_size[i] + 1) for i in range(self.nb_action_layers - 1)])
        self.mutation_dna_length = sum([mutation_nn_size[i+1]*(mutation_nn_size[i] + 1) for i in range(self.nb_mutation_layers - 1)])
        self.dna_length = self.action_dna_length + self.mutation_dna_length

        self.update_list() #crée l'adn, possible que si le mutation nn a été spécifié


    def mutate(self):  #clairement pas opti de créer la liste adn séparée, mais tellement plus pratique...
        new_dna = []
        for i in range(self.dna_length//self.mutation_nn_size[0]):
            adn_section = torch.FloatTensor([self.dna[i*self.mutation_nn_size[0]:(i+1)*self.mutation_nn_size[0]]])
            new_dna.extend((adn_section + self.mutation_nn(adn_section))[0].tolist())
            if random.rand() < self.mutation_nn_size[0]/self.dna_length:
                new_dna[i*self.mutation_nn_size[0] + int(random.rand()*self.mutation_nn_size[0])] += (random.random() - .5)*.5

        old_dna_end = self.dna[-(self.dna_length%self.mutation_nn_size[0]):] + [0.]*(self.mutation_nn_size[0]-(self.dna_length%self.mutation_nn_size[0]))
        new_dna_end = self.mutation_nn(torch.FloatTensor([old_dna_end]))[0].tolist()

        new_dna.extend(new_dna_end[:self.dna_length%self.mutation_nn_size[0]])            

        self.dna = list(new_dna)
        self.update_networks()
        
        
    def update_networks(self):  
        id = 0
        with torch.no_grad():
            for l in range(self.nb_action_layers - 1):
                
                self.action_nn.linear_relu_stack[2*l].weight = torch.nn.parameter.Parameter(torch.Tensor([[self.dna[id + j*self.action_nn_size[l] + i] for i in range(self.action_nn_size[l])] for j in range(self.action_nn_size[l+1])]))
                id += self.action_nn_size[l] * self.action_nn_size[l+1]
                self.action_nn.linear_relu_stack[2*l].bias = torch.nn.Parameter(torch.Tensor([self.dna[id + j] for j in range(self.action_nn_size[l+1])]))
                id += self.action_nn_size[l+1]
            
            for l in range(self.nb_mutation_layers - 1):
                self.mutation_nn.linear_relu_stack[2*l].weight = torch.nn.Parameter(torch.Tensor([[self.dna[id + j*self.mutation_nn_size[l] + i] for i in range(self.mutation_nn_size[l])] for j in range(self.mutation_nn_size[l+1])]))
                id += self.mutation_nn_size[l] * self.mutation_nn_size[l+1]
                self.mutation_nn.linear_relu_stack[2*l].bias = torch.nn.Parameter(torch.Tensor([self.dna[id + j] for j in range(self.mutation_nn_size[l+1])]))
                id += self.mutation_nn_size[l+1]
        

    def update_list(self):        
        temp_dna = []

        for l in range(self.nb_action_layers - 1):
            for w in self.action_nn.linear_relu_stack[2*l].weight:
                temp_dna.extend(w.tolist())
            temp_dna.extend(self.action_nn.linear_relu_stack[2*l].bias.tolist())
        
        for l in range(self.nb_mutation_layers - 1):
            for w in self.mutation_nn.linear_relu_stack[2*l].weight:
                temp_dna.extend(w.tolist())
            temp_dna.extend(self.mutation_nn.linear_relu_stack[2*l].bias.tolist())

        self.dna = temp_dna
        

    def evaluate_fitness(self, data): #devrait etre dans une class environnement ayant pour attributs des especes... mais pas d'interaction inter individus ici donc ok
        self.action_nn.eval()
        mean_dev = 0
        with torch.no_grad():
            pred = self.action_nn(data[0])
            mean_dev += ((pred-data[1])**2).type(torch.float).sum().item()   
        return mean_dev




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_gen', type=int,
                    help='Number of generations', default=100)

    parser.add_argument('--no_evo', dest='evolution', action='store_const',
                        const=False, default=True,
                        help='disables evolution process')

    parser.add_argument('--test', dest='test', action='store_const',
                        const=True, default=False,
                        help='lets you test the results !')

    args = parser.parse_args()


    
    if args.evolution:
        Boulier = Species(15, [2, 15, 1], [5, 20, 5])
        specimen, accuracy = Boulier.evolve(args.nb_gen, 50)
        action_nn, mutation_nn = specimen.action_nn, specimen.mutation_nn
        print(f"Error of best specimen = {accuracy}")
        acc_str = str(int(100*accuracy))
        action_name = "action_" + str(specimen.action_nn_size) + "_" + acc_str[0] + "." + acc_str[1:] + ".pth"
        torch.save(action_nn.state_dict(), action_name)
        mutation_name = "mutation_" + str(specimen.mutation_nn_size) + "_" + acc_str[0] + "." + acc_str[1:] + ".pth"
        torch.save(mutation_nn.state_dict(), mutation_name)

    # model = NeuralNetwork([2, 15, 1])
    # model.load_state_dict(torch.load("ElCalculator.pth"))

    # if args.test:

    #     x = input()
    #     while x != "q":
    #         a, b = x.split(" ")
    #         print(f"{float(a)} x {float(b)} = {model(torch.Tensor([[float(b), float(a)]]))[0][0]}.")
    #         x = input()


