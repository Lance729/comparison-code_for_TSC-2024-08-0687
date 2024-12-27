"""
Project: TaCo
File: latency_calculater.py
Description: This module is used to calculate the latency of the system.

Author:  Lance
Created: 2024-12-19
Email: lance.lz.kong@gmail.com
"""


from graphlize import get_graph
from get_network_state import Network_State_Generator
import numpy as np  



def calculate_latency(actor_allocation_scheme: np.ndarray, net_detail: dict, model_details: dict): # Input transmission scheme, calculate total latency.
    '''
    Description: This function is used to calculate the latency of the system.
    Args:
        actor_allocation_scheme: The actor_allocation_scheme for current state.
        net_detail: The network details.
        model_details: The model details.
    Returns:
        latency_total: The total latency of the system.

    '''
    # actor_allocation_scheme = np.array([[3], [4], [0], [0], [7]], np.int8) # Assume a segmentation scheme.
    '''The elements of the segmentation scheme here should represent the number of nodes. When making actor decisions, we should add 1 to the extracted index. Because if we start counting from 0, then the 0 that does not bear the amount of computation cannot be distinguished from the first node with an index of 0. Therefore, the element given by the scheme means the amount of computation of the number of nodes, and 0 means not bearing the amount of computation. Suppose an element is 3, then it represents the amount of computation within the third segmentation point. Since the calculation starts from zero, the amount of computation of the third node is sum(self.inference_graph.x[0:3,0]), which means the sum of the first 3 nodes.
    '''
    comp_latency = 0
    comm_latency = 0
    latency_total = 0
    comp_requirement = 0
    # comm_Kbits = 0
 

    net_graph, inference_graph = get_graph(net_detail, model_details)
    num_net_node, network_dim = net_graph.x.shape[0], net_graph.x.shape[1] # The number of network node features, currently 1
    num_inference_node, inference_dim = inference_graph.x.shape[0], inference_graph.x.shape[1] # The number of segmentation points and point features of the inference model. The segmentation point is also the action space. Currently, there are 8 segmentation points, considering the input and output layers, and the number of point features is 1

    # Create matrices for computation latency and communication latency
    comp_latency_matrix = np.zeros((num_net_node, 1))
    comm_latency_matrix = np.zeros((num_net_node, 1))

    # print("Segmentation scheme:", actor_allocation_scheme)
    for i in range(num_net_node): # The input model segmentation strategy, the elements in the strategy represent the number of segmentation points, not the index starting from 0.
        # print('Node %d'%(i+1))
        comp_ability = net_graph.x[i, 0] # The computing power of the node, taking the i-th element in the first column of x
        if i == 0:
            comm_speed = net_graph.edge_attr[i, 0]
            if actor_allocation_scheme[i] == 0:
                comp_requirement = 0
                comm_Kbits = inference_graph.edge_attr[0,0]
            elif 0 < actor_allocation_scheme[i] < num_inference_node: # If a non-first node i bears the amount of computation, calculate the amount of computation of the first actor_allocation_scheme[i] nodes and the output of this point, then clear the segmentation point and the previous ones.

                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-1] # The index of the number of nodes should be subtracted by 1
                inference_graph.x[0:actor_allocation_scheme[i], 0] = 0 # Set the selected points to zero
            elif actor_allocation_scheme[i] == num_inference_node: # If the segmentation point is the last one, calculate the previous amount of computation. The output is the output of the last segmentation point, but the last segmentation point has no output itself, it outputs the result, so the output is the same as the previous segmentation point.
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-2] # The transmission amount here should be subtracted by 2, because there is one less edge than the node itself. After subtracting 1 from the last node, the index is still exceeded, so it should be subtracted by 2.

        elif 0 < i < (num_net_node-1):  # When i is neither the first nor the last node
            comm_speed = net_graph.edge_attr[i, 0]
            if actor_allocation_scheme[i] == 0: # If a non-first node does not bear the amount of computation, the transmission data amount is the same as the last time, acting as a relay.
                comp_requirement = 0
                comm_Kbits = comm_Kbits
            elif 0 < actor_allocation_scheme[i] < num_inference_node: # If a non-first node i bears the amount of computation, calculate the amount of computation of the first actor_allocation_scheme[i] nodes and the output of this point, then clear the segmentation point and the previous ones.
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-1] # The index of the number of nodes should be subtracted by 1

                inference_graph.x[0:actor_allocation_scheme[i], 0] = 0 # Set the selected points to zero
            elif actor_allocation_scheme[i] == num_inference_node: # If the segmentation point is the last one, calculate the previous amount of computation. The output is the output of the last segmentation point, but the last segmentation point has no output itself, it outputs the result, so the output is the same as the previous segmentation point.
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-2] # The transmission amount here should be subtracted by 2, because there is one less edge than the node itself. After subtracting 1 from the last node, the index is still exceeded, so it should be subtracted by 2.
            if type(comm_Kbits) is not float : comm_Kbits = comm_Kbits.item()
        elif i == (num_net_node-1): # If i is the last node, it has no transmission amount or transmission speed, but it can bear the amount of computation. However, there is no need to clear the previous amount of computation.
            comm_speed = 0

            if actor_allocation_scheme[i] == 0: # If the last node does not bear the amount of computation, the transmission data amount is 0
                comp_requirement = 0
                comm_Kbits = 0 # The transmission amount of the last node is always 0
            # elif 0< actor_allocation_scheme[i] < num_inference_node: # If the last node i bears the amount of computation, calculate the amount of computation of the first actor_allocation_scheme[i] nodes
            elif actor_allocation_scheme[i] != 0: # If the last node i bears the amount of computation, calculate the amount of computation of the first actor_allocation_scheme[i] nodes
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                # inference_graph.x[0:actor_allocation_scheme[i], 0] = 0 # Set the selected points to zero

                comm_Kbits = 0 # The transmission amount of the last node is always 0

        
        # comp_latency[i] = comp_requirement/comp_ability
        # comm_latency[i] = comm_Kbits / (comm_speed+1e-10)
        comp_latency_now = comp_requirement/comp_ability
        comm_latency_now = comm_Kbits/(comm_speed+1e-10)
        latency_now = comp_latency_now+comm_latency_now # The latency of the current node is the sum of computation latency and communication latency

        comp_latency += comp_latency_now # comm_latency unit is ms, here the computing power of the node is GFlops, but the amount of computation is MFLOPs. Normally, the computing power should be multiplied by 1000 to get seconds, but we need milliseconds, so we don't convert the computing power.
        comm_latency += comm_latency_now # comm_latency unit is ms, here the transmission amount is KB, but the amount of computation is MBps. Normally, comm_Kbits should be divided by 1000 to get seconds, but we need milliseconds, so we don't convert comm_Kbits.

        # Store computation latency and communication latency into matrices
        comp_latency_matrix[i] = comp_latency_now
        comm_latency_matrix[i] = comm_latency_now

        # print('The latency of node %d is %f ms, including computation latency: %f ms, computation amount: %f MFLOPs, computing power: %f GFlops. Communication latency: %f ms, transmission amount: %f KBits, transmission speed: %f MBps'%(i+1, latency_now, comp_latency_now, comp_requirement, comp_ability, comm_latency_now, comm_Kbits, comm_speed))

    # print(comp_latency, comm_latency)
    # latency_total = np.sum(comp_latency) + np.sum(comm_latency)
    latency_total = comp_latency+comm_latency
    # print('Total latency: %f ms, computation latency: %f ms, communication latency: %f ms.'%(latency_total, comp_latency, comm_latency))
    return latency_total, comp_latency, comm_latency, comp_latency_matrix, comm_latency_matrix

# if __name__ == '__main__':
    
#     pass