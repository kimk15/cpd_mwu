from helper import rand_init
from helper import sample
from helper import bern
from helper import update_weights
from helper import update_factors
from helper import residual_error
from helper import norm
from helper import unfold

import numpy as np
import time
"""
Computes CPD for 3 dimensional tensors.
Returns A,B,C
"""
def CPD_MWU(X, F, sketching_rates, lamb, eps, nu, rank, num_iterations=100):
    # Keep residual errors + res time
    error = []
    res_time = 0

    # Cache norm + Id + unfolding of X
    norm_x = norm(X)
    Id = np.eye(rank)
    X_unfold = [unfold(X, mode=0), unfold(X, mode=1), unfold(X, mode=2)]

    # Initialize weights
    weights = np.array([1] * len(sketching_rates)) / (len(sketching_rates))

    # Randomly initialize A,B,C
    dim_1, dim_2, dim_3 = X.shape
    A, B, C = rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)

    # Append initialization residual error
    error.append(residual_error(X_unfold[0], norm_x, A,B,C))

    # Run CPD_MWU for num iterations
    for i in range(num_iterations):
        # Select sketching rate with probability proportional to w_i
        s = sample(sketching_rates, weights)

        # Solve Ridge Regression for A,B,C
        A, B, C = update_factors(A, B, C, X_unfold, Id, lamb, s, rank)

        # Update weights
        if bern(eps) == 1 and len(sketching_rates) > 1:
            update_weights(A, B, C, X_unfold, Id, norm_x, lamb, weights, sketching_rates, rank, nu, eps)

        print("iteration:", i)
        start = time.time()
        error.append(residual_error(X_unfold[0], norm_x, A,B,C))
        end = time.time()
        res_time += end-start
    return A,B,C, np.array(error), res_time

