"""
Project: TaCo
File: run.py
Description: This file runs other modules to reduce the latency.

Author:  Lance
Created: 2024-12-26
Email: lance.lz.kong@gmail.com
"""


import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from src import env_handler, reinforcement_learning, comparision_modules, configuration, utils
from env_handler import Environment_Handler
from reinforcement_learning import RL_for_DNN, RL_for_BayesianOptimization
from configuration import Network_Parameters
from dataclasses import dataclass, field
import time, json


@dataclass
class Parameters_DNN(Network_Parameters):
    """
    Extend network and energy consumption parameter class, add new attributes.
    """
    input_size: int = 1  # Input feature dimension
    hidden_sizes: list = field(default_factory=lambda: [44, 128, 64])  # Number of nodes in hidden layers
    output_size: int = 1  # Output offloading strategy
    learning_rate = 0.01
    max_timesteps = 1
    batch_size = 64
    num_episodes = 100
    gamma = 0.95  # Discount factor

@dataclass
class Parameters_for_BO(Network_Parameters):
    """
    Extend network and energy consumption parameter class, add new attributes.
    """
    input_size: int = 1  # Input feature dimension
    hidden_sizes: list = field(default_factory=lambda: [44, 128, 64])  # Number of nodes in hidden layers
    output_size: int = 1  # Output offloading strategy
    learning_rate = 0.01
    max_timesteps = 1
    batch_size = 64
    num_episodes = 10
    gamma = 0.95  # Discount factor

def RL_with_DNN():
    hyperparameters = Parameters_DNN()
    # myevn = Environment_Handler(hyperparameters)
    rl_impl = RL_for_DNN(hyperparameters)

    start_time = time.perf_counter()
    latencies = rl_impl.train()
    # Running program or code block
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Latencies: {latencies}, Elapsed Time: {elapsed_time}")  
    # latencies = latencies.tolist()
    results = {
        "latencies": latencies,
        "elapsed_time": elapsed_time
    }
    return results

def RL_with_BO():
    configuration = Parameters_for_BO()
    rl_impl = RL_for_BayesianOptimization(configuration)

    start_time = time.perf_counter()
    latencies = rl_impl.train()
    # Running program or code block
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Latencies: {latencies}, Elapsed Time: {elapsed_time}")
    latencies = [tensor.item() for tensor in latencies]
    results = {
        "latencies": latencies,
        "elapsed_time": elapsed_time
    }
    return results

    # return latencies, elapsed_time

if __name__ == '__main__':
    # RL_with_DNN()
    eval_DNN = True
    eval_BO = False

    if eval_DNN is True:
        DNN_impl_result = RL_with_DNN()
        with open('results/10DNN_impl_result.json', 'w') as f:
            json.dump(DNN_impl_result, f)
    if eval_BO is True:
        BO_impl_result = RL_with_BO()

        # Save dictionary as JSON file
        with open('results/BO_impl_result.json', 'w') as f:
            json.dump(BO_impl_result, f)

    print("All data has been saved to the results folder.")
