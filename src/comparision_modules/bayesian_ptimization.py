"""
Project: TaCo
File: bayesian_ptimization.py
Description: This module is used to optimize the hyperparameters of the model using Bayesian optimization.

Author:  Lance
Created: 2024-12-25
Email: lance.lz.kong@gmail.com
"""

import numpy as np
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.acquisition import gaussian_ei
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import torch




class BayesianOptimization:
    def __init__(self, objective_function, bounds, n_initial_points=5, kernel=None):
        """
        Initialize Bayesian optimizer
        :param objective_function: Objective function
        :param bounds: Parameter space boundaries
        :param n_initial_points: Number of initial sampling points
        :param kernel: Gaussian process kernel function
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.kernel = kernel if kernel else Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True)

        # Initialize sampling points
        self.X = np.random.uniform(0, 1, (self.n_initial_points, len(self.bounds)))
        self.y = np.array([self.objective_function(x) for x in self.X])

        # Fit initial Gaussian process model
        self.gp.fit(self.X, self.y)

    def expected_improvement(self, X):
        """
        Calculate Expected Improvement (EI)
        :param X: Set of sampling points to be evaluated
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        # Avoid numerical issues
        y_min = self.y.min()
        with np.errstate(divide='ignore'):
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0  # If the standard deviation is zero, EI is zero
        return ei

    def optimize(self, n_iter=20):
        """
        Perform Bayesian optimization
        :param n_iter: Maximum number of optimization iterations
        """
        for i in range(n_iter):
            # Find the next sampling point
            x_next_candidates = np.random.uniform(0, 1, (1000, len(self.bounds)))
            ei_values = self.expected_improvement(x_next_candidates)
            x_next = x_next_candidates[np.argmax(ei_values)]

            # Evaluate the objective function
            y_next = self.objective_function(x_next)

            # Update data
            self.X = np.vstack((self.X, x_next))
            self.y = np.append(self.y, y_next)

            # Update Gaussian process model
            self.gp.fit(self.X, self.y)

            print(f"Iteration {i+1}: Best Response = {self.y.min()}")

        # Return the optimal result
        best_config = self.X[np.argmin(self.y)]
        return best_config, self.y.min()




def objective_function(config):
    """
    Simulate an objective function, returning the response time (TmaxResp) for the given configuration.
    """
    x = np.array(config)
    noise = np.random.normal(0, 0.1)  # Add noise to simulate measurement error
    return np.sin(5 * x[0]) * (1 - np.tanh(x[1] ** 2)) + noise


class Bayesian_Optimization_for_RL:
    def __init__(self, bounds, num_samples=15):
        self.bounds = bounds  # Define variable ranges, e.g., [(1, 8), ..., (1, 8)]
        self.num_samples = num_samples
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))  # Gaussian process kernel
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6)
        self.X_sample = None  # Sampling points
        self.y_sample = None  # Corresponding objective values

    def sample(self, num_points=15):
        """Randomly generate initial sampling points, shape [num_points, dimension]"""
        samples = np.random.randint(1, 9, size=(num_points, len(self.bounds)))
        return torch.tensor(samples, dtype=torch.float32)

    def optimize_acquisition(self):
        """Maximize the acquisition function to select the next set of parameters"""
        def acquisition(x, xi=0.01):
            """UCB acquisition function"""
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            return mu + xi * sigma

        best_val = -np.inf
        best_x = None
        for _ in range(100):  # Multiple random start optimizations
            x0 = np.random.randint(1, 8, len(self.bounds))
            res = minimize(lambda x: -acquisition(x), x0, bounds=self.bounds, method='L-BFGS-B')
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x
        return torch.tensor(np.round(best_x).astype(int), dtype=torch.float32)

    def update(self, X_new, y_new):
        """Update sampling points"""
        # If X_new is a one-dimensional array, convert it to a two-dimensional array
        X_new = X_new.reshape(-1, 1)  # Convert to a two-dimensional array, shape (n_samples, 1)
        y_new = y_new.reshape(-1, 1)  # If y_new is also a one-dimensional array, it also needs to be converted to a two-dimensional array

        if self.X_sample is None:
            self.X_sample = X_new
            self.y_sample = y_new
        else:
            # Merge new data
            self.X_sample = np.vstack((self.X_sample, X_new))
            self.y_sample = np.vstack((self.y_sample, y_new))

        self.gp.fit(self.X_sample, self.y_sample)

    def propose_next(self):
        """Propose the next set of parameters"""
        proposals = [self.optimize_acquisition().numpy() for _ in range(self.num_samples)]
        return torch.tensor(proposals, dtype=torch.float32)


# def main():
#     # Parameter space
#     bounds = [(0.0, 1.0), (0.0, 1.0)]

#     # Initialize Bayesian optimizer
#     optimizer = BayesianOptimization(objective_function, bounds, n_initial_points=5)

#     # Run optimization
#     best_config, best_response = optimizer.optimize(n_iter=20)

#     print(f"Optimal configuration: {best_config}")
#     print(f"Minimum response time: {best_response}")

# if __name__ == '__main__':
#     main()
