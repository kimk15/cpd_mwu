%reload_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import tensorly.random as tl_rand
import tensorly.kruskal_tensor as tl_kruskal
import numpy as np

from timeit import default_timer as timer
from cpd_mwu import CPD_MWU
from data_gen import generate_collinear_factors


# Set up
congruence = 0.9
lamb = 0.001
shape = (300,300,300)
nu = 2
rank = 20
num_iterations = 100000
eps = 1/num_iterations

# Sketching rates
sketching_rates = list(np.linspace(10**(-3), 10**(-1), 4)) + [1]

# Generate random latent factors
F = np.array(tl_rand.random_kruskal(shape=shape, rank=rank, full=False, random_state=np.random.RandomState(seed=0)))
X = tl_kruskal.kruskal_to_tensor(F)

# Generate ill conditioned factors
F_ill = generate_collinear_factors(shape, rank, congruence)
X_ill = tl_kruskal.kruskal_to_tensor(F_ill)

# Run experiment for sketching with weight update
sketching_rates = list(np.linspace(10**(-3), 10**(-1), 4)) + [1]
start = timer()
A,B,C, error, res_time = CPD_MWU(X, F, sketching_rates, lamb, eps, nu, rank, num_iterations)
end = timer()
# Print out total time
print("Total time", end-start)
print("CPD time", end-start-res_time)
print("Residual error time", res_time)


# Save data