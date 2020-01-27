import tensorly as tl
import tensorly.random as tl_rand
import tensorly.base as tl_base
import tensorly.kruskal_tensor as tl_kruskal
import numpy as np
import matplotlib.pyplot as plt

from cpd_mwu import CPD_MWU

# Set up
X = tl_rand.random_kruskal((366,366,100), 5, full=True)
sketching_rates = list(np.linspace(10**(-3), 10**(-1), 4)) + [1]
lamb = 0.001
eps = 0.05 # 1/N where N = # interations
nu = 2
rank = 5  
num_iterations = 25

# Run experiment
A,B,C, error, diff_norm = CPD_MWU(X, sketching_rates, lamb, eps, nu, rank, num_iterations)

# Save error
with open("error.txt", "wb") as fp:
	pickle.dump(error, fp)

with open("diff_norm.txt", "rb") as fp:
	pickle.dump(diff_norm, fp)

# Plot out error
x = [i for i in range(len(error))]
plt.title('Residual error vs # iterations')
plt.plot(x, error)