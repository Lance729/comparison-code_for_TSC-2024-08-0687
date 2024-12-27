"""
Project: TaCo
File: env_generator.py
Description: This model is used to generate the environment of the system. It includes the network state, device state, graph generator, and energy calculator.

Author:  Lance
Created: 2024-12-20
Email: lance.lz.kong@gmail.com
"""

from get_network_state import Network_State_Generator
from energy_calculator import Energy_Calculator
from graphlize import get_graph
from latency_calculator import calculate_latency
from utils import save_data_to_json, read_json_file
import numpy as np
from dataclasses import dataclass
import torch    
from torch.distributions import Categorical

@dataclass
class Network_Parameters:
    """
    Network and energy parameters class, used to define the characteristics of device computation and communication.
    """
    # Device computation and node configuration
    UE_flops: float = 12.0  # User equipment computation power (GFlops)
    
    num_node: int = 15  # Number of network nodes
    tau: int = 4  # ES computation power is tau times that of UE
    ES_flops: float =UE_flops * tau  # Edge server computation power (GFlops)
    network_type: str = '4G+5G'  # Network type
    dispersion: float = 0.1  # Random dispersion of network rate
    seed: int = 42  # Random seed
    inference_model_index: tuple = ('VGG16', 'Cifar100')  # Inference model and dataset

    # Energy parameters
    UE_comp_power: float = 6.0  # User equipment computation power (W)
    # ES_comp_power: float = 10.0  # Edge server computation power (W)
    UE_transmission_power: float = 20.0  # User equipment transmission power (mW)
    ES_transmission_power: float = 100.0  # Access point transmission power (mW)
    transmission_efficiency: float = 0.9  # Data transmission efficiency (0~1)
    computation_energy_per_gflop: float = 0.1  # Energy required per GFlops computation (J)

    comp_efficiency = UE_flops / UE_comp_power  # Computation efficiency \mu = GFlops / W
    ES_comp_power = UE_flops * tau / comp_efficiency  # Edge server computation power (W)

class Environment_Handler:
    # Define class attributes
    energy_calculator = None
    network_state = None

    def __init__(self, params):
        # Initialize class attributes
        Environment_Handler.energy_calculator = Energy_Calculator(params)
        Environment_Handler.network_state = Network_State_Generator(params)


    @classmethod
    def get_next_network_state(cls, if_with_energy=False, **kwargs):
        '''Get the next network state.'''

        # Update other parameters passed to the network_state object
        for key, value in kwargs.items():
            if hasattr(cls.network_state, key):
                setattr(cls.network_state, key, value)

        #* ================Get the next network state================
        if if_with_energy: # Get the next network state with energy.
            net_detail_with_energy, model_details_with_energy, transmission_power_attributes, comp_power_attributes = cls.energy_calculator.get_device_feature_with_energy()
            print(f"The network state with energy is generated successfully!")
            return net_detail_with_energy, model_details_with_energy
        else: # Get the next network state without energy.
            net_detail, model_details = cls.network_state.get_net_device_feature()
            return net_detail, model_details


    @classmethod
    def get_latency_results(cls, actor_allocation_scheme, network_states, model_details) -> tuple: 
        '''
        Description: Get latency results.
        Args:
            actor_scheme: Transmission scheme.
            network_states: Network state.
        return: latency_results = (latency_total, comp_latency, comm_latency, comp_latency_matrix, comm_latency_matrix) Computation and communication latency of all nodes, and other related latency results, need to be unpacked when used.
        '''
        # if network_states or model_details is None:
        #     network_states, model_details = cls.get_next_network_state()
    
    
        latency_results = calculate_latency(actor_allocation_scheme, network_states, model_details)
        return latency_results
    
    @classmethod
    def get_graph(cls, net_detail, model_details):
        '''Get network graph.'''
        net_graph, inference_graph = get_graph(net_detail, model_details)
        return net_graph, inference_graph

    @classmethod
    def get_energy_consumption(cls, 
                               actor_allocation_scheme, 
                               network_states = None, 
                               model_details = None, 
                               transmission_latency_matrix_with_energy = None, 
                               comp_latency_matrix_with_energy = None
                               ) -> tuple: 
        '''
        Description: Get energy consumption results.
        Args:
            actor_scheme: Transmission scheme.
            network_states: Network state.
        return: energy_results = (energy_total, comp_energy, comm_energy, comp_energy_matrix, comm_energy_matrix) Computation and communication energy consumption of all nodes, and other related energy results, need to be unpacked when used.
        '''
        if_with_energy = True
        if network_states or model_details is None:
            network_states, model_details = cls.get_next_network_state(if_with_energy)

        if transmission_latency_matrix_with_energy or comp_latency_matrix_with_energy is None:
            latency_results = cls.get_latency_results(actor_allocation_scheme, network_states, model_details)
            transmission_latency_matrix_with_energy = latency_results[4]
            comp_latency_matrix_with_energy = latency_results[3]

            
        env_state_with_energy_results = cls.energy_calculator.get_device_feature_with_energy() # -> tuple[net_detail, model_details, transmission_power_attributes, comp_power_attributes]

        transmission_power_attributes = env_state_with_energy_results[2]
        comp_power_attributes = env_state_with_energy_results[3]
        total_energy_results = cls.energy_calculator.get_total_energy(
            transmission_power_ALLdevices = transmission_power_attributes, # all devices' transmission power
            computation_power_ALLdevices = comp_power_attributes, # all devices' computation power
            transmission_latency_set=transmission_latency_matrix_with_energy, # all devices' communication latency
            comp_latency_set=comp_latency_matrix_with_energy # all devices' computation latency
        )
        print("Get energy consumption successfully!")
        return total_energy_results # -> tuple[energy_total, comp_energy, comm_energy, comp_energy_matrix, comm_energy_matrix]




    def actor_normalize(self, actor_predict, num_net_node, num_inference_node, indicator): # Normalize the raw segmentation scheme into the expected segmentation scheme.
        '''Input the log probability form of the actor distribution given by the original GNN, indicator is 1 for sampling by probability, 0 for sampling by maximum value, output the organized action actor, and the log probability of each action.
        input: log probability actor_predict, sampling indicator
        output: organized action actor, and log probability of each action.
        '''
        if indicator == 1: # sample() samples according to probability, each sample is different
            a_distrib = Categorical(torch.exp(actor_predict))  # size([5,8]), torch.exp(actor_predict) does not change the value of actor_predict
            actor_class = a_distrib.sample() # Get the raw classification, still need to organize size([5])
        else: # indicator is 0, which means sampling by the maximum value of each row, the result is the same for the same input.
            actor_class = torch.max(actor_predict,1).indices # torch.max function returns two tensors when the parameter is 1, values tensor is the maximum value in each row, indices is the index of the maximum value in each row.

        pre = [] # Selected samples, which are the original probabilities, not after log.
        for j in range(num_net_node):
            pre.append(actor_predict[j, actor_class[j]].item()) #size([5]) because there are five points
        # np.array(pre)
        pre = torch.tensor(pre)

        actor_class = np.array(actor_class)
        # print(np.array(actor_class))
        actor = np.zeros(num_net_node, dtype=np.int8)
        for i in range(num_net_node): # Loop to assign model segmentation points to each node
            if i==0:
                actor[i] = (actor_class[i]+1) # Directly assign when assigning to the first point
            elif i==1: # The second point is assigned only if it is larger than the first point, otherwise it is 0
                actor[i] = (actor_class[i]+1) * ( (actor_class[i]>actor_class[i-1]) )
            elif 1 < i < (num_net_node-1): # From the third point to the second last point, judge whether it has appeared before
                actor[i] = (actor_class[i]+1) * ( 0 not in (actor_class[i]>actor_class[0:i]) ) # Intermediate nodes are set to 0 if they are less than or equal to all previous nodes
            elif i==num_net_node-1: # If none of the previous nodes choose the last segmentation point, the last node must choose to bear the final computation.
                actor[i] = num_inference_node * ((num_inference_node-1) not in actor_class)
            # print("Node %d, choose segmentation point %d"%(i+1, actor[i]))

        return actor, pre





    def reward(self, latency_old, latency_now):
        '''Calculate reward, we plan to use the reward calculation method based on latency:
        reward = ( l^{i-1} / l^{i} ) * ( l^{i-1} - l^{i} ) # Should try taking an absolute value
        The advantage of this formula is that we hope the latency generated by the next action is less than the previous latency, so there is a subtraction,
        When the latency increases, the reward will be negative, making the overall smaller. The ratio in front also indicates that we hope the subsequent latency is smaller, making the reward inversely proportional to the latency.
        input: latency_old, latency_now. latency_later is the previous latency, latency_now is the current action latency
        output: return reward
        '''
        # reward = ( latency_old / latency_now ) * ( 2*latency_old -  latency_now )
        reward = 1 / latency_now

        # if latency_now<latency_old:
        #   reward = 100
        # elif latency_now>latency_old:
        #   reward = -10
        # elif latency_now == latency_old:
        #   reward = 10
        return torch.log(reward)*10
    
    @classmethod
    def evn_reset(cls)-> tuple: # tuple(net_graph, inference_graph)
        '''Environment reset function, currently it seems to be the input network graph and inference graph
        input: no input
        output: initial state of the network self.net_graph, initial state of the inference task VGG16_graph.
        '''
        observation = cls.get_next_network_state() # -> tuple[net_detail, model_details]
        reset_observation = cls.get_graph() #-> tuple(net_graph, inference_graph)
        # reset_observation = reset_net_graph, cls.inference_graph
        return reset_observation

    def evn_step(self,action, latency_old): # Return the new environment observation state, current action reward and done
        '''Environment update function, input action, return the next system state next_observation, reward and done. This done is a boolean value, 1 means the game is over.
        input: action,
        output: next_observation, reward, done

        Regarding next_observation, my plan is that the input of GNN is self.net_graph, VGG16_graph, and the shape of this input cannot be changed.
        But after inputting an action, that is, after selecting a segmentation point for the next node, in self.net_graph, VGG16_graph, the graph nodes and edges before the selected point are all zero
        For example, in the above network and VGG model, if we assign the third segmentation point to the third node, the returned next_observation, the points before the third node in self.net_graph are all zero, and the points before the third segmentation point in VGG16_graph are also set to zero.
        This can also ensure that the action space is continuously reduced!
        '''
        # print('latency_old=', latency_old)
        latency_new, comp_latency_new, comm_latency_new = self.calculate_latency(action)
        latency_dic = {'latency_new': latency_new, 'comp_latency_new': comp_latency_new, 'comm_latency_new':comm_latency_new}
        # print('latency_new=', latency_new)
        reward = self.reward(latency_old, latency_new)
        # print('reward=', reward)
        # next_observation = self.get_graph() # Put the graph in a list, next_observation[0] represents the first graph
        next_net_graph, self.inference_graph = self.get_graph()
        next_observation = next_net_graph, self.inference_graph

        done = 1

        return next_observation, latency_new, latency_dic, reward, done

    def loss_function(self, latency):
        '''Establish loss function
        output: loss, respectively actor and critic loss
        '''
        # latency = cauculate_latency(segment)
        aloss = 1/latency
        return aloss    


def display_env_usage():
    """
    Function to demonstrate the usage of Environment_Handler.
    """
    scheme = [1,7,8]
    network_params = Network_Parameters(
        num_node=len(scheme),    # Number of nodes
        tau=6                    # Computation power of edge server is tau times that of UE
    )
    env = Environment_Handler(network_params)
    energy_results = env.get_energy_consumption(scheme)
    print(f"total_energy: {energy_results[0]},\n transmission_energy_consumption_mWs: {energy_results[1]},\n computation_energy_consumption_ws*1000: {energy_results[2]},\n trans_energy_set: {energy_results[3]},\n comp_energy_set: {energy_results[4]}")
    print('The environment is generated successfully!')

# if __name__ == '__main__':


#     scheme = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0]

#     network_params = Network_Parameters(
#         num_node = len(scheme),    # Number of nodes
#         tau= 6                     # Computation power of edge server is tau times that of UE
#     )

#     env = Environment_Handler(network_params)
#     # latency_results = env.get_latency_results(scheme_test)
#     energy_results = env.get_energy_consumption(scheme)
#     # (total_energy, transmission_energy_consumption_mWs, computation_energy_consumption_ws*1000, trans_energy_set, comp_energy_set)
#     print(f"total_energy: {energy_results[0]},\n transmission_energy_consumption_mWs: {energy_results[1]},\n computation_energy_consumption_ws*1000: {energy_results[2]},\n trans_energy_set: {energy_results[3]},\n comp_energy_set: {energy_results[4]}")
#     # #* ==========Save data
#     # save_data_to_json(file_path = '/home/lance/code_taco/results/energy_results.json', 
#     #                   new_data = energy_results, 
#     #                   top_level_key= f"energy_consumption_results")
#     print('The environment is generated successfully!')
