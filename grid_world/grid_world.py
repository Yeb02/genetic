import pygame
import numpy as np
from numpy import random, array
from grid_agent import Species, Specimen
import torch
import os 
from perlin_numpy import generate_perlin_noise_2d


class Grid():
    def __init__(self, height, width, value_init=False, list_init=False, random_init=False, value_for_init=None):    
        self.contains_list = list_init
        if list_init:
            self.values = [[[] for _ in range(width)] for _ in range(height)]
        elif random_init:    
            # self.values = array([[random.rand() for _ in range(width)] for _ in range(height)])
            if height%10!=0 or width%10!=0:
                print("GRID SIZE MUST BE MULTIPLE OF 4")
            self.values = (1+generate_perlin_noise_2d((height, width), (4, 4)))/2

        elif value_init:
            self.values = array([[value_for_init for _ in range(width)] for _ in range(height)])
        self.height, self.width = height, width
    
    # def flattened_square(self, center_x, center_y, radius):
    #     square = self.values[center_x - radius:center_x+radius+1, center_y - radius:center_y+radius+1]
    #     square_flattened = []

    #     for line in square:
    #         square_flattened.extend(line)

    #     return square_flattened

    def flattened_square(self, center_x, center_y, radius):
        square_flattened = []
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                square_flattened.append(self.values[(center_x+i)%self.height][(center_y+j)%self.width])

        return square_flattened


class Game():
    def __init__(self, height, width, nb_species, specimens_base_per_species, no_save=False):
        self.height = height
        self.width = width
        self.radius = 1
        self.initial_energy = 50
        self.time_between_matings = 15

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        self.nn_size = [(2*self.radius+1)**2 * 3 + 5, 20, 6]
        self.species = [Species(int(specimens_base_per_species*(specimens_base_per_species-1)/2), self.nn_size, self.device, spe_i, height, width,
                        self.initial_energy, self.time_between_matings) for spe_i in range(nb_species)]
        self.nb_species = nb_species
        

        self.ind_nb_threshold = specimens_base_per_species

        self.reset(no_save, has_ancestors=False)
        self.elite = [[species.specimens[i] for i in range(self.ind_nb_threshold)] for species in self.species]

        """
        L'entrée du nn est, dans un carré de (2*radius+1) ** 2, les données des food, paint, player grids, 
         l' énergie de l'individu considéré, son x, son y, 2 réels de mémoire.
        La sortie du nn est dans l'intervalle 0 1:
        - S'il faut peindre (>.5 alors oui)
        - La valeur de la peinture s'il y a lieu
        - La valeur en x du déplacement. Comme la sortie est une sigmoide, on prend le signe de int((x-.5)*V) pour que les deux directions soient symétriques
        V est un parametre qui dépend de la répartition autour de .5
        - Idem, pour y. 
        - Les deux valeurs de mémoire.
        """

    def reset(self, no_save=True, has_ancestors=True):
        self.paint_grid = Grid(self.height, self.width, value_init=True, value_for_init=0.)
        self.food_grid = Grid(self.height, self.width, random_init=True)
        self.player_grid = Grid(self.height, self.width, list_init=True)
        self.player_id_grid = Grid(self.height, self.width, value_init=True, value_for_init=-1.)
        self.nb_species_under_threshold = [0 for _ in range(self.nb_species)]
        self.mating_occurences = 0
        self.init_species(has_ancestors, no_save)

    def init_species(self, has_ancestors, no_save):
        if has_ancestors:
            print("breeding .....")
            for spe_i,species in enumerate(self.species):
                species.nb_specimens = int(self.ind_nb_threshold*(self.ind_nb_threshold-1)/2)
                species.specimens = [Specimen(self.nn_size, spe_i, self.device,
                                  1 + int(random.rand() * (self.height-2)),
                                  1 + int(random.rand() * (self.width-2)), ind_i,
                                  self.initial_energy, self.time_between_matings)
                                  for ind_i in range(species.nb_specimens)]
                
                self.breed_from_elite(spe_i)

        elif not no_save:
            try:
                self.elite = [[] for _ in range(self.nb_species)]
                absolute_path = str(self.nb_species)+"_spe_"+str(int(self.ind_nb_threshold*(self.ind_nb_threshold-1)/2))+r"_ind/"
                for spe_i,species in enumerate(self.species):
                    self.elite[spe_i] = [species.specimens[j] for j in range(self.ind_nb_threshold)]
                    for j,specimen in enumerate(self.elite[spe_i]):
                        name = "specimen_"+str(spe_i)+"_"+str(j)+".pth"
                        specimen.nn.load_state_dict(torch.load(absolute_path+name))
                    self.breed_from_elite(spe_i)
                print("Used the found save.")
            except:
                print("No save found. Have you checked that species nbr and ind per species nbr correspond ?")
        else:
            print("You chose not to use any of the available previous saves.")

        for species in self.species:
            species.reset(int(self.ind_nb_threshold*(self.ind_nb_threshold-1)/2))
            for specimen in species.specimens:
                self.player_grid.values[specimen.x][specimen.y].append([specimen.ind_indice, specimen.spe_indice])
                self.player_id_grid.values[specimen.x][specimen.y] = specimen.spe_indice

    
    def breed_from_elite(self, spe_i):
        for i in range(self.ind_nb_threshold):
            for j in range(i+1, self.ind_nb_threshold):

                for l in range(len(self.nn_size) -1):
                    for w in range(self.nn_size[l]*self.nn_size[l+1]):   #weights loop
                        with torch.no_grad():
                            if random.random() < .5:
                                v = self.elite[spe_i][i].nn.linear_relu_stack[2*l].weight[w//self.nn_size[l]][w%self.nn_size[l]]
                            else:
                                v = self.elite[spe_i][j].nn.linear_relu_stack[2*l].weight[w//self.nn_size[l]][w%self.nn_size[l]]
                            self.species[spe_i].specimens[i].nn.linear_relu_stack[2*l].weight[w//self.nn_size[l]][w%self.nn_size[l]] = v
                    for b in range(self.nn_size[l+1]):   #bias loop
                        with torch.no_grad():
                            if random.random() < .5:
                                self.species[spe_i].specimens[i].nn.linear_relu_stack[2*l].bias[b] = self.elite[spe_i][i].nn.linear_relu_stack[2*l].bias[b]
                            else:
                                self.species[spe_i].specimens[i].nn.linear_relu_stack[2*l].bias[b] = self.elite[spe_i][j].nn.linear_relu_stack[2*l].bias[b]

                self.species[spe_i].specimens[i].mutate(2/self.species[spe_i].specimens[i].dna_length)


    def specimen_acting(self, specimen):
        actions = specimen.take_actions(self.paint_grid.flattened_square(specimen.x, specimen.y, self.radius),
                                        self.food_grid.flattened_square(specimen.x, specimen.y, self.radius),
                                        self.player_id_grid.flattened_square(specimen.x, specimen.y, self.radius),
                                        specimen.x/self.height , specimen.y/self.width, specimen.energy/self.initial_energy)
        if actions[0] > .5:
            self.paint_grid.values[specimen.x][specimen.y] = actions[1]
        specimen.memory = actions[4:]
                        
        return np.sign(int(20*(actions[2] - .5))), np.sign(int(20*(actions[3] - .5)))  #dx, dy


    def remove_specimen_from_case(self, specimen):
        for i,l in enumerate(self.player_grid.values[specimen.x][specimen.y]):
            if l[1] == specimen.spe_indice and l[0] == specimen.ind_indice:
                del self.player_grid.values[specimen.x][specimen.y][i]
                break
        if len(self.player_grid.values[specimen.x][specimen.y]) == 0 :
            self.player_id_grid.values[specimen.x][specimen.y] = -1


    def on_arriving_to_case(self, specimen, species):
        id_to_delete = []
        
        self.player_grid.values[specimen.x][specimen.y].append([specimen.ind_indice, specimen.spe_indice])

        for i,l in enumerate(self.player_grid.values[specimen.x][specimen.y][:-1]):

            #mating ?
            if l[1] == specimen.spe_indice and specimen.time_before_mating_possible==species.specimens[l[0]].time_before_mating_possible==0: 
                child = species.mate(specimen, species.specimens[l[0]])
                self.mating_occurences+=1
                self.player_grid.values[specimen.x][specimen.y].append([child.ind_indice, child.spe_indice])
            
            #fighting ?
            if l[1] != specimen.spe_indice:
                if specimen.energy > self.species[l[1]].specimens[l[0]].energy:   #combat gagné
                    specimen.energy += self.species[l[1]].specimens[l[0]].energy 
                    self.species[l[1]].kill_specimen(l[0])    
                    id_to_delete.append(i)
                    
                else:  #combat perdu
                    self.species[l[1]].specimens[l[0]].energy += specimen.energy
                    self.remove_specimen_from_case(specimen)
                    species.kill_specimen(specimen.ind_indice)  
                    # pour que meurent vraiment les vaincus.
                    for id in id_to_delete[::-1]: 
                        del self.player_grid.values[specimen.x][specimen.y][id]

                    break
                
        if specimen.time_before_mating_possible != 0:
            specimen.time_before_mating_possible -= 1
        self.player_id_grid.values[specimen.x][specimen.y] = specimen.spe_indice
        specimen.energy += self.food_grid.values[specimen.x][specimen.y]
        self.food_grid.values[specimen.x][specimen.y] = 0.

        for id in id_to_delete[::-1]: #du plus grand au plus petit
            del self.player_grid.values[specimen.x][specimen.y][id]


    def draw(self, window, presentation_mode):
        for species in self.species:
            for specimen in species.specimens:
                if specimen is not None:
                    if presentation_mode:
                        pygame.draw.rect(window, species.color, pygame.Rect(specimen.x*10, specimen.y*10, 10, 10))
                    else:
                        window.set_at((specimen.x, specimen.y), species.color)


    def time_step(self):
        #TODO replenish food grid ?

        for spe_i,species in enumerate(self.species):
            
            #picking elite
            if self.nb_species_under_threshold[spe_i]==0 and  species.nb_specimens <= self.ind_nb_threshold: 
                self.nb_species_under_threshold[spe_i] = 1
                temp_elite = [specimen for specimen in species.specimens if specimen is not None]
                temp_elite.extend(self.elite[spe_i][:-len(temp_elite)])
                self.elite[spe_i] = temp_elite

            for specimen in species.specimens:
                #dead specimens are skipped (None slots in the specimens list)
                if specimen is None:
                    continue
                #dies if energy depleted
                if specimen.energy < 0:
                    self.remove_specimen_from_case(specimen)
                    species.kill_specimen(specimen.ind_indice)
                    continue
                
                # specimen choosing actions, and returning choosen movements.
                dx, dy = self.specimen_acting(specimen)

                if dx != 0 or dy != 0:  #moving this turn
                    specimen.energy -= .3
                    
                    #remove specimen from the case it occupied last turn
                    self.remove_specimen_from_case(specimen)

                    # moving specimen
                    specimen.x = (dx + specimen.x)%self.height
                    specimen.y = (dy + specimen.y)%self.width
                    # places specimen on the grid case it moves to and possibly interacts with specimens living on it
                    self.on_arriving_to_case(specimen, species)
                    

                else:  #immobile this turn
                    specimen.energy -= .25
        
        return sum(self.nb_species_under_threshold) < self.nb_species

    def save_game(self):
        absolute_path = str(self.nb_species)+"_spe_"+str(int(self.ind_nb_threshold*(self.ind_nb_threshold-1)/2))+r"_ind/"
        for u,elite in enumerate(self.elite):
            for v,specimen in enumerate(elite):
                path =  absolute_path + "specimen_"+str(u)+"_"+str(v)+".pth"
                torch.save(specimen.nn.state_dict(), path)

    def run_game(self, graphics=False, presentation_mode=False, nb_steps=5):
        try:
            os.mkdir(str(self.nb_species)+"_spe_"+str(int(self.ind_nb_threshold*(self.ind_nb_threshold-1)/2))+r"_ind")
        except:
            print("There are existing saves for this configuration. Running the game will overwrite them.")

        if graphics:
            pygame.init()
            if presentation_mode:
                window = pygame.display.set_mode((self.height*10, self.width*10))
            else:
                window = pygame.display.set_mode((self.height, self.width))
            pygame.display.set_caption("Evolution")
            self.running = True
        
        graphics = False

        for i in range(nb_steps):
            if i%5==3:
                self.save_game()
                print("saved")
            if i > 900:
                graphics = True
                presentation_mode = True

            j = 0
            while self.time_step():
                j+=1
                if graphics:
                    if presentation_mode:
                        pygame.time.delay(30) 
                    pygame.time.delay(1) 
                    window.fill((0,0,0))
                    for event in pygame.event.get(): 
                        if event.type == pygame.QUIT: 
                            self.running = False  
                    keys = pygame.key.get_pressed()  
                    if keys[pygame.K_p]:
                        self.pause()
                    self.draw(window, presentation_mode)
                    pygame.display.update()

            print(f"env  {i},  {j} steps,  {self.mating_occurences}  mating occurences")
            self.reset()   

        if graphics:
            pygame.quit()  

    def pause(self):
        paused = True
        while paused:
            pygame.time.delay(20)
            keys = pygame.key.get_pressed()  
            if keys[pygame.K_P]:
                paused = False



if __name__ == "__main__":
    the_game = Game(height=20, width=20, nb_species=2, specimens_base_per_species=5, no_save=False)
    the_game.run_game(nb_steps=1000, graphics=True)