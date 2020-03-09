import matplotlib.pyplot as plt
import tensorly.random as tl_rand
import tensorly.kruskal_tensor as tl_kruskal
import numpy as np
from timeit import default_timer as timer

from cpd_mwu import CPD_MWU

# Set up
shape = (300, 300, 300)
lamb = 0.001
eps = 0.05 # 1/N where N = # interations
nu = 2
rank = 5  
num_iterations = 1000
F = np.array(tl_rand.random_kruskal(shape=shape, rank=rank, full=False, random_state=np.random.RandomState(seed=0)))
X = tl_kruskal.kruskal_to_tensor(F)

# Run experiment for sketching with weight update
sketching_rates = list(np.linspace(10**(-3), 10**(-1), 4)) + [1]
start = timer()
A,B,C, error, res_time = CPD_MWU(X, F, sketching_rates, lamb, eps, nu, rank, num_iterations)
end = timer()
# Print out total time
print("Total time", end-start)
print("CPD time", end-start-res_time)
print("Residual error time", res_time)



# Plot out error
x = range(len(error))
plt.title('Residual error vs # iterations')
plt.xlabel('Iterations')
plt.ylabel('Residual Error')
plt.plot(x, error, label='No sketch + weight update')
plt.figure()