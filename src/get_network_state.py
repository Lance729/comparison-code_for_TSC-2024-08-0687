import numpy as np
from dataclasses import dataclass
# from env_handler import Network_Parameters


class Network_State_Generator:
    def __init__(self, config_para):
            config_parameter = config_para
            self.UE_flops = config_para.UE_flops
            self.num_node = config_para.num_node
            self.tau = config_para.tau
            self.network_type = config_para.network_type
            self.dispersion = config_para.dispersion
            self.seed = config_para.seed
            self.inference_model_index = config_para.inference_model_index
    def get_network_status(self):
        """
        Get network status.
        
        Returns:
        np.ndarray: Network status matrix.
        """
        # Set random seed
        np.random.seed(self.seed)
        
        # Define speeds for each network type
        upspeed = {
            '4G+5G': 7.5,
            '5G': 21.8,
            '4G': 7.5,
            '3G': 1.5
        }
        downspeed = {
            '4G+5G': [175.3, 22.1],
            '5G': [175.3, 175.3],
            '4G': [22.1, 22.1],
            '3G': [4.9, 4.9]
        }

        # Generate base speed matrix
        speed_matrix = np.array([[1] for _ in range(self.num_node-1)], float)
        speed_matrix[0] = upspeed[self.network_type]
        speed_matrix[1:,] = downspeed[self.network_type][0]
        speed_matrix[-1] = downspeed[self.network_type][1]

        # Generate normal distribution speed matrix based on dispersion
        speed_matrix_noisy = np.random.normal(speed_matrix, self.dispersion, (self.num_node-1, 1))

        # Limit the range of the speed matrix
        if self.network_type == '4G+5G':
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] < 1.5] = 1.5
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] > 18.2] = 18.2
            speed_matrix_noisy[1:-1, :][speed_matrix_noisy[1:-1, :] < 98] = 98
            speed_matrix_noisy[1:-1, :][speed_matrix_noisy[1:-1, :] > 447.8] = 447.8
            speed_matrix_noisy[-1, :][speed_matrix_noisy[-1, :] < 11.9] = 11.9
            speed_matrix_noisy[-1, :][speed_matrix_noisy[-1, :] > 55.7] = 55.7
        elif self.network_type == '5G':
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] < 7.5] = 7.5
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] > 66] = 66
            speed_matrix_noisy[1:, :][speed_matrix_noisy[1:, :] < 98] = 98
            speed_matrix_noisy[1:, :][speed_matrix_noisy[1:, :] > 447.8] = 447.8
        elif self.network_type == '4G':
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] < 1.5] = 1.5
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] > 18.2] = 18.2
            speed_matrix_noisy[1:, :][speed_matrix_noisy[1:, :] < 11.9] = 11.9
            speed_matrix_noisy[1:, :][speed_matrix_noisy[1:, :] > 55.7] = 55.7
        elif self.network_type == '3G':
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] < 1] = 1
            speed_matrix_noisy[0, :][speed_matrix_noisy[0, :] > 3] = 3
            speed_matrix_noisy[1:, :][speed_matrix_noisy[1:, :] < 2] = 2
            speed_matrix_noisy[1:, :][speed_matrix_noisy[1:, :] > 8] = 8

        return speed_matrix_noisy

    def get_net_device_feature(self)->tuple:
        '''Get network information and inference model data, return their respective dictionaries. Here, the network environment has fixed and non-fixed differences, network_index represents which wireless network environment, model_index represents which model and which dataset,
        The computing power of Apple's M1 chip is 36.845GFlops, Apple's A14 terminal chip is 12.960GFlops, my computer is i5-8257U, the computing power is 14.689GFlops, and the inference based on Cifar100's VGG16 on my computer is about 50ms, theoretically 45ms, this slowness is due to I/O interface and other reasons.
        The computing power of GTX1080Ti is 10.6TFlops=10600GFlops, in previous literature, the simulation used ImageNet, which requires much more computing power.

        input: evn_index represents a fixed environment, model_index is to choose which model and dataset, such as model_index = ('VGG16','Cifar100')

        The model graph and network graph should be set separately.
        '''

        node_feature = np.array([[self.UE_flops] for _ in range(self.num_node)], float) # Generate a matrix based on UE computing power, in the form of (num_node, 1)
        node_feature[1:-1] = self.UE_flops * self.tau # Change the values between the second and the second last to AP computing power, AP computing power equals UE * multiplier
        # node_feature = np.array([[18], [40], [45], [50], [20]]) # Here is GFlops, size=(num_node, 1)
        node_edge_index = list([(i,i+1) for i in range(self.num_node-1)]) # This is the adjacency matrix. Generate a list similar to list([(0,1), (1, 2), (2, 3), ……,(3, num_node-1)]). If num_node=5, it is list([(0,1), (1, 2), (2, 3), (3, 4)])

        # Use the get_network_status function to get CSI
        node_edge_features = self.get_network_status()

        assert len(node_edge_index) == len(node_edge_features), "The length of the wireless network adjacency matrix and edge features are not equal, they should be equal."
        node_edge_features = np.array(node_edge_features).reshape(self.num_node-1,1) # Convert network speed to column matrix, in the form of (num_node-1,1)


        '''Above is the wireless network graph, below is the inference model graph'''

        if self.inference_model_index[0] == 'VGG16':
            name_layer = np.array([['Input'],
                        ['block2_conv1'],
                        ['block3_conv1'],
                        ['block3_conv3'],
                        ['block4_conv2'],
                        ['block5_conv1'],
                        ['softmax'],
                        ['result']]).reshape(8,1) # Extract layer names and organize them into the form of 1*number of layers. Each layer is a class.
            if self.inference_model_index[1] == 'Cifar100':
                comp_Mflops = np.array([[0], [117.01939], [113.33192], [151.03276], [113.28276], [94.38843], [76.31415], [0]]) # Extract the MFLOPs of each layer and organize them into the form of 1*number of layers.
                inference_edge = list([(0,1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]) # This method is used to construct an unweighted adjacency matrix, k=1 is to move the diagonal up one row.
                commu_kbit = np.array([[98.304], [2097.152], [524.288], [524.288], [262.144], [65.536], [3.2]])  # Extract the Kbits output of each layer and organize them into the form of 1*(number of layers-1).

            elif self.inference_model_index[1] == 'ImageNet':
                comp_Mflops = np.array([[0], [5734.65], [5554.02282], [7400.83102], [5551.60564], [4625.74317], [2097.25185], [0]]) # Extract the MFLOPs of each layer and organize them into the form of 1*number of layers.
                inference_edge = list([(0,1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]) # This method is used to construct an unweighted adjacency matrix, k=1 is to move the diagonal up one row.
                commu_kbit = np.array([[4816.896], [51380.224], [25690.112], [25690.112], [12845.056], [3211.264], [32]])  # Extract the Kbits output of each layer and organize them into the form of 1*(number of layers-1).

        assert len(inference_edge) == len(commu_kbit), "The length of the inference model network adjacency matrix and edge features are not equal, they should be equal."
        net_detail = {'net_node_feature':node_feature,
                        'net_node_edge_index':node_edge_index,
                        'net_node_edge_features':node_edge_features}
        model_details = {'name_layer': name_layer,
                            'inference_edge': inference_edge,
                            'comp_Mflops': comp_Mflops,
                            'commu_kbit': commu_kbit}

        return net_detail, model_details





# # Example usage. Already tested
# if __name__ == '__main__':
#     UE_flops = 12.960  # Example
#     num_node = 5
#     tau = 2.0  # Example
#     network_type = '4G+5G'
#     dispersion = 0.1
#     seed = 42
#     generator = Network_State_Generator(UE_flops, num_node, tau, network_type, dispersion, seed)
#     network_status = generator.get_network_status()
#     print(network_status)
#     net_detail, model_details = generator.get_net_device_feature()
#     print(net_detail)
#     print(model_details)
