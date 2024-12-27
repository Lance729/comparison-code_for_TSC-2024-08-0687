"""
Project: TaCo
File: reinforcement_learning.py
Description: This module defines the reinforcement learning modules used in the system. 

Author:  Lance
Created: 2024-12-25
Email: lance.lz.kong@gmail.com
"""

from env_handler import Environment_Handler
from comparision_modules import DNN_modules, bayesian_ptimization, lstm_based_s2s
import torch.optim as optim
from scipy.stats import loguniform
from dataclasses import dataclass, field
from configuration import Network_Parameters
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

@dataclass
class Parameters(Network_Parameters):
    """
    Extend network and energy consumption parameter class, add new attributes.
    """
    input_size: int = 1  # Input feature dimension
    hidden_sizes: list = field(default_factory=lambda: [44, 128, 64])  # Number of hidden layer nodes
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
    hidden_sizes: list = field(default_factory=lambda: [44, 128, 64])  # Number of hidden layer nodes
    output_size: int = 1  # Output offloading strategy
    learning_rate = 0.01
    max_timesteps = 1
    batch_size = 64
    num_episodes = 100
    gamma = 0.95  # Discount factor

class RL_impl():
    """
    This class mainly uses RL to train GCN and can accept externally passed models for flexible use.
    """
    def __init__(self, 
                 CSI_dataset=None,  # CSI data
                 actor_model=None,  # Externally passed actor model
                 target_model=None  # Externally passed target model
                ):
        # self.CSIset = CSI_dataset
        self.actor_model = actor_model
        # self.target_model = target_model
        self.low = 1e-4  # Lower limit of learning rate range
        self.high = 1e-2  # Upper limit of learning rate range

        # if self.actor_model is None:
        #     raise ValueError("Actor and target models must be provided")

    @classmethod
    def get_actor_model(cls, config: Parameters , model_name: str= 'DNN'):
        """Get actor model"""

        if model_name == 'DNN':
            model = DNN_modules.TaskOffloadingDNN(config.input_size, config.hidden_sizes, config.output_size)

        
        return model


    def update_target(self, tau):
        """Update the parameters of the target model"""
        for param_tensor in self.actor_model.state_dict():
            self.target_model.state_dict()[param_tensor].data.copy_(
                (1 - tau) * self.target_model.state_dict()[param_tensor].data +
                tau * self.actor_model.state_dict()[param_tensor].data
            )

    def train(self, num_net_node, ue_flops, ap_times_ue, speed, rand, inference_model, gamma, tau, episode):
        """
        Main process of training using reinforcement learning
        """
        G_t = 0
        rewards = []
        score = {
            'reward_list': [],
            'latency_list': [],
            'actor_list': [],
            'actor_loss_list': []
        }

        # Initialize environment
        myevn = Get_evn_station(
            number_node=num_net_node,
            CSI_dataset=self.CSIset,
            UE_flops=ue_flops,
            tau=ap_times_ue,
            network_speed=speed,
            rand_index=rand,
            inference_model_index=inference_model
        )
        observation_reset = myevn.evn_reset() # -> tuple[net_graph, inference_graph]
        net_graph, inference_graph = observation_reset[0], observation_reset[1]
        num_net_node, network_dim = net_graph.x.shape[0], net_graph.x.shape[1]
        num_inference_node, inference_dim = inference_graph.x.shape[0], inference_graph.x.shape[1]

        # Initial action and latency
        actor_predict, ac_value = self.actor_model(observation_reset)
        actor, actor_probability = myevn.actor_normalize(
            actor_predict, num_net_node, num_inference_node, indicator=0
        )
        latency_old = myevn.calculate_latency(actor)

        # Training loop
        for i in range(episode):
            learning_rate = loguniform.rvs(self.low, self.high)
            optimizer = optim.RMSprop(self.actor_model.parameters(), lr=learning_rate)

            next_obs, latency_new, latency_dic, reward, done = myevn.evn_step(actor, latency_old)
            actor_predict, ac_value = self.actor_model(next_obs)
            actor, actor_probability = myevn.actor_normalize(
                actor_predict, num_net_node, num_inference_node, indicator=1
            )

            reward = reward.item()
            score['latency_list'].append(latency_new)
            score['reward_list'].append(reward)
            score['actor_list'].append(actor)
            rewards.append(reward)

            target_value = self.target_model(next_obs)
            TD = reward + gamma * target_value
            TD_error = (-ac_value + TD)
            critic_loss = F.smooth_l1_loss(input=ac_value, target=TD)
            actor_loss = (-actor_probability * TD_error.detach()).mean()
            score['actor_loss_list'].append(actor_loss.item())

            loss = actor_loss - critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.update_target(tau)
            latency_old = latency_new

        return score, latency_dic, self.actor_model, actor_loss



class RL_for_DNN(RL_impl):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.env_handler = Environment_Handler(config)
        self.num_inference_node = None
    def get_DNN_model(self, input_size, model_name: str= 'DNN'):
        """Get actor model"""
        model = DNN_modules.TaskOffloadingDNN()
        # self.actor_model = model
        return model



    def _get_network_state_in_one_dim(self, net_detail, model_details)-> torch.Tensor:
        """
        Extract the values of specified keys and organize them into a one-dimensional PyTorch Tensor.
        
        Parameters:
            net_detail (dict): Dictionary containing network details.
            model_details (dict): Dictionary containing model details.
        
        Returns:
            torch.Tensor: Merged 1D Tensor.
        """
        # Extract the values of specified keys
        net_node_feature = net_detail['net_node_feature']
        net_node_edge_features = net_detail['net_node_edge_features']
        name_layer = model_details['comp_Mflops']
        commu_kbit = model_details['commu_kbit']
        self.num_inference_node = len(name_layer)
        # Ensure all values are Torch Tensors
        tensors = [
            torch.tensor(net_node_feature).flatten(),
            torch.tensor(net_node_edge_features).flatten(),
            torch.tensor(name_layer).flatten(),
            torch.tensor(commu_kbit).flatten()
        ]
        
        # Concatenate all Tensors into a 1D Tensor
        result_tensor = torch.cat(tensors)
        
        return result_tensor.float()

    def train(self):
        model = self.get_DNN_model(self.config.input_size)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        gamma = self.config.gamma  # Discount factor
        latencies = []
        model.train()
        for episode in range(self.config.num_episodes):
            observation = self.env_handler.get_next_network_state()
            state = self._get_network_state_in_one_dim(observation[0], observation[1])
            state = state.view(1, -1)  # Ensure state shape is [1, 44]
            print(f"State shape: {state.shape}")  # Debug print
            episode_rewards = []
            log_probs = []


            for t in range(self.config.max_timesteps):
                # Get action probability distribution
                action_predict = model(state)  # Model output
                

                # Generate actual actions using environment methods
                action, action_predict = self.env_handler.actor_normalize(
                    action_predict, self.config.num_node, self.num_inference_node, indicator=1
                )
                print(f"Action action: {action}")  # Debug print
                # Calculate reward and latency
                latency = self.env_handler.get_latency_results(action,observation[0], observation[1])
                latency_value = latency[0]  # Extract the appropriate value from the tuple
                reward = -latency_value  # Use negative latency as reward

                # Record reward and log_prob
                # dist = torch.distributions.Categorical(action_predict)
                log_prob = action_predict
                log_probs.append(log_prob)
                episode_rewards.append(reward)
                latencies.append(latency_value)

                # Update state
                next_observation = self.env_handler.get_next_network_state()
                next_state = self._get_network_state_in_one_dim(next_observation[0], next_observation[1])
                next_state = next_state.view(1, -1)  # Ensure next_state shape is [1, 44]
                state = next_state

                # # Termination condition
                # if self.env_handler.is_terminal_state():
                #     break

            # Calculate discounted rewards
            total_reward = sum(episode_rewards)
            discounted_rewards = self._compute_discounted_rewards(episode_rewards, gamma)
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            # Ensure log_prob is a tensor that requires gradients
            log_prob = action_predict  # Model output
            log_prob = log_prob.requires_grad_()  # Force enable gradient calculation
            # Calculate policy_loss
            policy_loss = []
            for log_prob, reward in zip(log_probs, discounted_rewards):
                # Ensure reward also has gradients enabled
                reward = torch.tensor(reward).float()  # Force reward to be a float tensor
                policy_loss.append(-log_prob * reward)

            policy_loss = torch.cat(policy_loss).sum()  # Sum all losses

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            # Convert total_reward to a Python numeric type
            total_reward_value = total_reward.item() if isinstance(total_reward, torch.Tensor) else total_reward

            # If latencies is a list containing tensors, use flatten or item to convert them
            latencies_flat = [latency_value.item() if isinstance(latency_value, torch.Tensor) else latency_value for latency_value in latencies]

            # Calculate mean
            latencies_mean = np.mean(latencies)

            # Print results
            print(f"Episode {episode+1}/{self.config.num_episodes}, Total Reward: {total_reward_value:.4f}, Avg Latency: {latencies_mean:.4f}")
        return latencies_flat

    def _compute_discounted_rewards(self, rewards, gamma):
        """Calculate discounted rewards"""
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards


class RL_for_BayesianOptimization:
    def __init__(self, config, **kwargs):
        self.config = config
        self.env_handler = Environment_Handler(config)
        self.bayes_optimizer = bayesian_ptimization.Bayesian_Optimization_for_RL(bounds=[(1, 8)] , num_samples=15)  # Define optimizer boundaries
        self.num_inference_node = 8

    def _get_network_state_in_one_dim(self, net_detail, model_details) -> torch.Tensor:
        """Same as the previous method, process network state into one-dimensional input"""
        # Extract and concatenate state information
        tensors = [
            torch.tensor(net_detail['net_node_feature']).flatten(),
            torch.tensor(net_detail['net_node_edge_features']).flatten(),
            torch.tensor(model_details['comp_Mflops']).flatten(),
            torch.tensor(model_details['commu_kbit']).flatten()
        ]
        return torch.cat(tensors).float()

    def _reformulact_action(self, actor_class):
        # self.config.num_node, self.num_inference_node
        actor_class = np.array(actor_class)
        # print(np.array(actor_class))
        actor = np.zeros(self.config.num_node, dtype=np.int8)
        for i in range(self.config.num_node): # Loop to assign model split points to each node
            if i==0:
                actor[i] = (actor_class[i]+1) # Directly assign when assigning to the first point
            elif i==1: # The second point is assigned only if it is larger than the first point, otherwise it is 0
                actor[i] = (actor_class[i]+1) * ( (actor_class[i]>actor_class[i-1]) )
            elif 1 < i < (self.config.num_node-1): # From the third point to the second last point, check if it appears before
                actor[i] = (actor_class[i]+1) * ( 0 not in (actor_class[i]>actor_class[0:i]) ) # Intermediate nodes are set to 0 if they are less than or equal to all previous nodes
            elif i==self.config.num_node-1: # If the last split point is not chosen by all previous nodes, the last node must choose to bear the final computation.
                actor[i] = self.num_inference_node * ((self.num_inference_node-1) not in actor_class)
            # print("Node %d chooses split point %d"%(i+1, actor[i]))
        return actor

    def train(self):
        gamma = self.config.gamma
        latencies = []

        for episode in range(self.config.num_episodes):
            observation = self.env_handler.get_next_network_state()
            state = self._get_network_state_in_one_dim(observation[0], observation[1])
            state = state.view(1, -1) 

            episode_rewards = []

            # Bayesian optimization proposes initial actions
            bayes_action = self.bayes_optimizer.propose_next()
            

            for t in range(self.config.max_timesteps):
                # Simulate real environment actions
                action = bayes_action  # Replace part of RL strategy
                action = self._reformulact_action(action)
                print(f"Bayes Suggested Action: {action}")

                
                latency = self.env_handler.get_latency_results(action, observation[0], observation[1])
                latency_y = latency[3]+latency[4]
                latency_value = latency[0]  # Latency value
                reward = -latency_value

                episode_rewards.append(reward)
                latencies.append(latency_value)

                # Update Bayesian optimization data
                self.bayes_optimizer.update(action, latency_y)

                # Update state
                next_observation = self.env_handler.get_next_network_state()
                next_state = self._get_network_state_in_one_dim(next_observation[0], next_observation[1])
                next_state = next_state.view(1, -1)
                state = next_state

            # Calculate discounted rewards
            discounted_rewards = self._compute_discounted_rewards(episode_rewards, gamma)
            print(f"Episode {episode+1}, Total Reward: {sum(episode_rewards).item():.4f}")

        return latencies

    def _compute_discounted_rewards(self, rewards, gamma):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards




# def RL_DNN():

#     configuration = Parameters()
#     myevn = Environment_Handler(configuration)
#     observation_reset = myevn.get_next_network_state() # -> tuple[net_detail, model_details]
    
#     # # Training
#     rl_impl = RL_for_DNN(configuration)
#     # one_demission = rl_impl._get_network_state_in_one_dim(observation_reset[0], observation_reset[1])
#     # DNN_model = rl_impl.get_DNN_model()
#     # rl_impl = RL_impl(actor_model=DNN_nodel)
#     latencies = rl_impl.train()
#     print(latencies)  
#     print("Run successfully!")


# def RL_BO():
#     configuration = Parameters_for_BO()
#     rl_impl = RL_for_BayesianOptimization(configuration)
#     latencies = rl_impl.train()
#     print(latencies)
    
# if __name__ == '__main__':
#     RL_BO()