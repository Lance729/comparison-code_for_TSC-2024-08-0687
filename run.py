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
import argparse
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import env_handler, reinforcement_learning, comparision_modules, configuration, utils
from env_handler import Environment_Handler, display_env_usage
from reinforcement_learning import RL_for_DNN, RL_for_BayesianOptimization
from configuration import Network_Parameters
from dataclasses import dataclass, field


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
    rl_impl = RL_for_DNN(hyperparameters)

    start_time = time.perf_counter()
    latencies = rl_impl.train()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Latencies: {latencies}, Elapsed Time: {elapsed_time}")
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
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Latencies: {latencies}, Elapsed Time: {elapsed_time}")
    latencies = [tensor.item() for tensor in latencies]
    results = {
        "latencies": latencies,
        "elapsed_time": elapsed_time
    }
    return results


def env_handler_run():
    network_params = Network_Parameters(
                        num_node = 3,    # Number of nodes
                        tau= 6           # Computation power of edge server is tau times that of UE
                        )
    env_handler = Environment_Handler(network_params)
    net_detail, model_details = env_handler.get_next_network_statet()
    return net_detail, model_details

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RL algorithms for TaCo project.")
    parser.add_argument("--eval_DNN", action="store_true", help="Run the DNN-based RL implementation")
    parser.add_argument("--eval_BO", action="store_true", help="Run the Bayesian Optimization-based RL implementation")
    parser.add_argument("--eval_network_env", action="store_true", help="Run the network environment")

    args = parser.parse_args()

    display_env_usage()

    if args.eval_DNN:
        DNN_impl_result = RL_with_DNN()
        os.makedirs('results', exist_ok=True)
        with open('results/DNN_impl_result.json', 'w') as f:
            json.dump(DNN_impl_result, f)
        print("DNN results saved to results/DNN_impl_result.json")

    if args.eval_BO:
        BO_impl_result = RL_with_BO()
        os.makedirs('results', exist_ok=True)
        with open('results/BO_impl_result.json', 'w') as f:
            json.dump(BO_impl_result, f)
        print("BO results saved to results/BO_impl_result.json")


    if args.eval_network_env:
        net_detail, model_details = env_handler_run()
        print('The environment is generated successfully!')
        print(f"Network tranmission speed: {net_detail['net_node_feature']}, \n 
              Network devices computing ability: {net_detail['node_feature']}, \n
              Model inference edge: {model_details['inference_edge']}, \n
              Model computation workload Mflops: {model_details['comp_Mflops']}, \n
              Model communication workload kbit: {model_details['commu_kbit']}, \n
              "
             )


    if not args.eval_DNN and not args.eval_BO:
        print("No evaluation mode selected. Use --eval_DNN or --eval_BO to specify the mode.")
