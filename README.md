# Introduction

This project is designed to reduce latency in network environments using various reinforcement learning techniques.

## Baseline Details
The code of solutions that Nath's PTC [9] and Meng's DTOO [10] are listed in path `src/comparision_modules`. 

1. The PTC solution adopts Bayesian optimization (BO) as scheduler, which is built in `src/comparision_modules/bayesian_ptimization.py`

2. The DTOO solution adopts DNN as scheduler, which is built in `src/comparision_modules/DNN_modules.py`

## How to Run


1. The dependencies are managed by Conda. Ensure you have correctly installed the Conda and all the necessary dependencies installed. You can install them using:
    ```bash
    conda env create -f requirements.yml
    ```

2. To run the main program, execute:
    ```bash
    python3 run.py
    ```

## Demonstrating Environment Usage

The `display_env_usage` function in `env_handler.py` demonstrates how to use the `Environment_Handler` class. It initializes the environment and prints the energy consumption results.

To see this demonstration, simply run the main program as shown above. The output will include the energy consumption details and a confirmation message indicating that the environment was generated successfully.

