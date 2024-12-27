"""
Project: TaCo
File: energy_calculator.py
Description: Calculate the energy of taco

Author:  Lance
Created: 2024-12-15
Email: lance.lz.kong@gmail.com
"""

from get_network_state import Network_State_Generator
import numpy as np
from dataclasses import dataclass
from latency_calculator import calculate_latency # -> tuple[latency_total, comp_latency, comm_latency, comp_latency_matrix, comm_latency_matrix]


# FIXME: issue 2 - Calculation capability



@dataclass
class Network_Parameters:
    """
    Network and energy parameters class, used to define the characteristics of device computation and communication.
    """
    # Device computing power and node configuration
    UE_flops: float = 12.0  # User equipment computing power (GFlops)
    
    num_node: int = 0  # Number of network nodes
    tau: int = 4  # ES computing power is a multiple of UE
    ES_flops: float =UE_flops * tau  # Edge server computing power (GFlops)
    network_type: str = '4G+5G'  # Network type
    dispersion: float = 0.1  # Random dispersion of network rate
    seed: int = 42  # Random seed
    inference_model_index: tuple = ('VGG16', 'Cifar100')  # Inference model and dataset

    # Energy parameters
    UE_comp_power: float = 6.0  # User equipment computing power (W)
    # ES_comp_power: float = 10.0  # Edge server computing power (W)
    UE_transmission_power: float = 20.0  # User equipment transmission power (mW)
    ES_transmission_power: float = 100.0  # Access point transmission power (mW)
    transmission_efficiency: float = 0.9  # Data transmission efficiency (0~1)
    computation_energy_per_gflop: float = 0.1  # Energy required per GFlops (J)

    comp_efficiency = UE_flops / UE_comp_power  # Computing efficiency \mu = GFlops / W
    ES_comp_power = UE_flops * tau / comp_efficiency  # Edge server computing power (W)

class Energy_Calculator(Network_State_Generator):
    """
    Energy calculation class, inherited from Network_State_Generator.
    Provides methods for calculating communication and computation energy consumption.
    """

    def __init__(self, params: Network_Parameters):
        """
        Initialize the Energy_Calculator class.

        Parameters:
        - params (Network_Parameters): Instance of network and energy parameters.
        """
        super().__init__(params)
        if params.num_node == 0: raise ValueError("Number of nodes must not be zero.")
        self.params = params
        self.comp_power_set = None
        self.transmission_power_ALLdevices = None

    def get_device_feature_with_energy(self):
        '''Get network state, device information, device computation and transmission energy consumption, etc.
            Returns: net_detail_with_energy, model_details
            Where net_detail_with_energy is a dictionary containing the following key-value pairs:
            - 'net_node_feature': Node feature matrix, the first column is computing power, the second column is computing power
            - 'net_node_edge_features': Edge feature matrix, including communication rate between nodes and the second column is transmission power
            - 'transmission_power': Transmission power

        '''
        #! Test successful
        #? The information of the model seems to be useless
        net_detail_without_energy, model_details = super().get_net_device_feature()

        net_detail_with_energy = net_detail_without_energy

        #*===== Adding power attributes to the devices
        comp_power_attributes = np.full((self.params.num_node, 1), self.params.UE_comp_power) # Add computing power attributes, add a column to the node attributes
        comp_power_attributes[1:-1] = self.params.ES_comp_power # Assign power to ES. That is, change the elements between the second and the second to last to ES computing power, ES power = UE*tau/computing efficiency
        self.comp_power_set = comp_power_attributes
        net_detail_with_energy['net_node_feature'] = np.hstack((net_detail_with_energy['net_node_feature'], comp_power_attributes)) # Add computing power attributes, add a column to the node attributes

        #*===== Adding transmission power attributes to the edges
        # transmission_power_attributes = np.array([[self.params.UE_transmission_power] for _ in range(self.params.num_node - 1)], float)  # transmission power
        transmission_power_attributes = np.full((self.params.num_node - 1, 1), self.params.UE_transmission_power) # Add transmission power attributes, add a column to the edge attributes
        transmission_power_attributes[1:] = self.params.ES_transmission_power  # We assume that only the first edge is UE transmission power, assign ES transmission power to all elements except the first element in this matrix
        self.transmission_power_ALLdevices = transmission_power_attributes
        net_detail_with_energy['net_node_edge_features'] = np.hstack((net_detail_with_energy['net_node_edge_features'], transmission_power_attributes))

        return net_detail_with_energy, model_details, transmission_power_attributes, comp_power_attributes

    @classmethod
    def calculate_transmission_energy(cls, transmission_power_ALLdevices, transmission_latency_set=None, segment=None ):
        #TODO: Here we only need to know the transmission time, because we can assume that the channel quality, power, and speed are constant, so for a certain amount of data, the transmission time is fixed.

        """
        Calculate transmission energy consumption.

        Parameters:
        - data_kbits (float): Amount of data transmitted (Kbits).
        - transmission_power_ALLdevices (mW): Transmission power (mW).
        - transmission_latency_set: Communication latency set generated by all devices (s)
        Returns:
        - transmission_energy_joules: Transmission energy consumption (mWs), and the transmission energy consumption set of each device
        """
        # transmission_energy_set = np.array([])
        # transmission_energy_joules = 0.0        
        if transmission_latency_set is None: 
            if segment is None: raise ValueError("Either transmission_latency_set or segment must be provided.") # If transmission_latency_set is not provided, segment must be provided
            transmission_latency_set= calculate_latency(segment)[4] # Select the fourth element of the communication return result, which is the communication latency set

        # if len(transmission_latency_set) != len(transmission_power_ALLdevices) : raise ValueError("The length of transmission_latency_set and transmission_power_ALLdevices must be the same.") # If the lengths are inconsistent, throw an exception

        # FIXME: Calculate communication energy consumption. The transmission matrix transmission_power_ALLdevices here is actually one element less than the transmission latency matrix transmission_latency_set, because the latency matrix has an additional target node's 0 transmission latency. So we delete the last element, so below we check if the last element of transmission_energy_set[-1] is 0
        transmission_latency_set = transmission_latency_set[:-1] # Delete the last element
        transmission_energy_set = transmission_power_ALLdevices * transmission_latency_set  # Calculate the communication energy consumption of each device
        transmission_energy_joules = sum(transmission_energy_set) # Calculate total energy consumption mWs


        if transmission_energy_joules == 0.0 or transmission_energy_set.size == 0: 
             # If no energy consumption is calculated, or the last target node's transmission energy consumption is not 0, throw an exception
            raise ValueError("Energy calculation error: transmission_energy_joules and transmission_energy_set must be calculated.")
        return transmission_energy_joules, transmission_energy_set

    @classmethod
    def calculate_computation_energy(cls, computation_power_ALLdevices, computation_latency_set=None, segment=None):
        """
        Calculate computation energy consumption.

        Parameters:
·        - computation_mflops (float): MFlops required for computation.
         - device_computation_power_set (mW): Device computation power (mW).

        Returns:
        - float: Computation energy consumption (J), and the computation energy consumption set of each device
        """
        comp_energy_set = []
        comp_energy_joules = 0.0

        if computation_latency_set is None: 
            if segment is None: raise ValueError("Either transmission_latency_set or segment must be provided.") # If transmission_latency_set is not provided, segment must be provided
            results_m = calculate_latency(segment) # -> tuple[latency_total, comp_latency, comm_latency, comp_latency_matrix, comm_latency_matrix]
            computation_latency_set= results_m[3] # Select the fifth element of the communication return result, which is the computation latency set

        # FIXME: Calculate computation energy consumption
        comp_energy_set = computation_power_ALLdevices * computation_latency_set  # Calculate the computation energy consumption of each device
        comp_energy_joules = sum(comp_energy_set) # Calculate total energy consumption mWs

        if comp_energy_joules == 0.0 or comp_energy_set.size == 0:
             # If no energy consumption is calculated, or the last target node's computation energy consumption is not 0, throw an exception
            raise ValueError("Energy calculation error: comp_energy_joules and comp_energy_set must be calculated.")
        
        return comp_energy_joules, comp_energy_set

    @classmethod
    def get_total_energy(cls, 
                        transmission_power_ALLdevices,                         
                        computation_power_ALLdevices, 
                        transmission_latency_set=None,
                        comp_latency_set=None, 
                        segment=None
                         ):
        """
        Calculate the total energy consumption of the entire network (including transmission and computation).
        E_total = E_comm + E_comp
        where E_comm = sum(E_comm_i) and E_comp = sum(E_comp_i)
        E_comm_i = transmission_power_i * transmission_time_i
        E_comp_i = comp_power_i * comp_time_i

        input:
        - transmission_power_ALLdevices: Transmission power mW
        - computation_power_ALLdevices: Computation power W
        - transmission_latency_set: Transmission latency set mWs
        - comp_latency_set: Computation latency set Ws
        - segment: Scheduling scheme
        Returns:
        - float: Total energy consumption (mWs).
        """
        
        
        
        # Extract network characteristics
        # transmission_power_ALLdevices = net_detail['transmission_power']
        # comp_power_set = net_detail['comp_power']


        total_energy = 0.0
        # Calculate communication energy consumption
        transmission_energy_consumption_mWs, trans_energy_set = cls.calculate_transmission_energy(transmission_power_ALLdevices, transmission_latency_set)
        # Calculate computation energy consumption
        computation_energy_consumption_ws, comp_energy_set = cls.calculate_computation_energy(computation_power_ALLdevices, comp_latency_set)

        # Calculate total energy consumption of all nodes
        total_energy = sum(trans_energy_set) + sum(comp_energy_set)*1000 # Transmission energy consumption is mW/s, computation energy consumption is W/s, so computation energy consumption needs to be multiplied by 1000
        if total_energy != (transmission_energy_consumption_mWs + computation_energy_consumption_ws*1000): raise ValueError("Energy calculation error: total_energy does not match the sum of transmission_energy and computation_energy.")
        total_energy_results = (total_energy, transmission_energy_consumption_mWs, computation_energy_consumption_ws*1000, trans_energy_set, comp_energy_set)
        return total_energy_results


# def main():
#     """
#     Main function, execute energy calculation tasks.
#     """
#     # Create parameter instance
#     network_params = Network_Parameters(
#         num_node = 3,    # Number of nodes
#         tau= 4           # Computing power of edge server is tau times that of UE
#     )
#     segment_test = [1,7,8] 

#     # Create energy calculator instance
#     energy_calculator = Energy_Calculator(network_params)




#     # #** Get network and device features without energy (for testing, delete in release version)
#     # net_detail, model_details = energy_calculator.get_net_device_feature()
#     # latency_tuple = calculate_latency(segment_test, net_detail, model_details) # latency_total, comp_latency, comm_latency, comp_latency_matrix, comm_latency_matrix

#     # comm_latency_matrix = latency_tuple[4]    
#     # comp_latency_matrix = latency_tuple[3]

    
#     #** =======Get network and device features, latency, and energy with energy
#     env_state_with_energy_results = energy_calculator.get_device_feature_with_energy() # -> tuple[net_detail, model_details, transmission_power_attributes, comp_power_attributes]
#     net_detail_with_energy = env_state_with_energy_results[0]
#     model_details = env_state_with_energy_results[1]

#     #* ========Get various latency data
#     latency_with_energy = calculate_latency(segment_test, net_detail_with_energy, model_details)  # -> latency_total, comp_latency, comm_latency, comp_latency_matrix, comm_latency_matrix

#     comp_latency_matrix_with_energy = latency_with_energy[3]
#     comm_latency_matrix_with_energy = latency_with_energy[4]

#     # Check if latency matrices match (for testing, delete in release version)
#     # if np.array_equal(comm_latency_matrix, comm_latency_matrix_with_energy) is False: raise ValueError("Latency calculation error: comm_latency_matrix does not match comm_latency_matrix_with_energy.")

#     # #**============ Transmission energy consumption test successful
#     transmission_power_attributes = env_state_with_energy_results[2]
#     transmission_transmission_energy_joules, transmission_energy_set = energy_calculator.calculate_transmission_energy(
#         transmission_power_ALLdevices = transmission_power_attributes,  # all devices' transmission power
#         transmission_latency_set=comm_latency_matrix_with_energy    # all devices' communication latency
#         )
#     print(f"transmission_energy_set: {transmission_energy_set}, total: {transmission_transmission_energy_joules} mW/s \n")

#     #*=============== Computation energy consumption test successful
#     comp_power_attributes = env_state_with_energy_results[3]
#     comp_energy_joules, comp_energy_set = energy_calculator.calculate_computation_energy(
#         computation_power_ALLdevices=comp_power_attributes, # all devices' computation power
#         computation_latency_set=comp_latency_matrix_with_energy # all devices' computation latency
#         )
#     print(f"comp_energy_set: {comp_energy_set}, total: {comp_energy_joules} W/s \m")

#     #*=============== Total energy consumption test successful
#     total_energy_results = energy_calculator.get_total_energy(
#         transmission_power_ALLdevices = transmission_power_attributes, # all devices' transmission power
#         computation_power_ALLdevices = comp_power_attributes, # all devices' computation power
#         transmission_latency_set=comm_latency_matrix_with_energy, # all devices' communication latency
#         comp_latency_set=comp_latency_matrix_with_energy # all devices' computation latency
#     )
#     total_energy_consumption = total_energy_results[0].item()
#     print(f"total_energy_results: {total_energy_consumption} mWs \n")
#     print("All tests passed successfully!")
#     # print(model_details)
#     # # Calculate total energy consumption
#     # total_energy = energy_calculator.get_total_energy()
#     # print(f"Total energy consumption: {total_energy:.2f} J")


# if __name__ == '__main__':
#     main()
