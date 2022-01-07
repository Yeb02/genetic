import numpy as np
from numpy import random

p_vraie = 1
n_tests = 100
n_etapes_avg = 0

 
for _ in range(n_tests):
    vraie =  random.rand() * 100

    v0 = 50
    delta = 25

    n_etapes = 0
    while np.abs(v0-vraie) > 1: 
        if v0 > vraie:
            bigger = True
        else:
            bigger = False

        if random.rand() < p_vraie:
            pass
        n_etapes += 1

    n_etapes_avg += n_etapes

print(n_etapes_avg/n_tests)

