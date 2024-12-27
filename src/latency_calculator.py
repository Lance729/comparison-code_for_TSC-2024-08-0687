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



def calculate_latency(actor_allocation_scheme: np.ndarray, net_detail: dict, model_details: dict): # 输入传输方案，计算总延迟。
    '''
    Description: This function is used to calculate the latency of the system.
    Args:
        actor_allocation_scheme: The actor_allocation_scheme for current state.
        net_detail: The network details.
        model_details: The model details.
    Returns:
        latency_total: The total latency of the system.

    '''
    # actor_allocation_scheme = np.array([[3], [4], [0], [0], [7]], np.int8) # 假设一个分割方案。
    '''这里的分割方案元素应该代表着第几个节点，在actor决策的时候我们就应该让取出来的下标都+1。因为如果按照下标从0开始算的话，那么不承担计算量的0就无法与下标为0的第一个节点相区分。所以方案给出的元素是第几个节点的计算量的意思，0代表不承担计算量。假设某一个元素为3，那么他代表了包含第三个分割点以内的计算量。而所以计算量的时候，第三个节点的下标是2。
    因为从零开始计算，所以第三个节点的计算量为sum(self.inference_graph.x[0:3,0])，这个表示前3个节点相加，
    '''
    comp_latency = 0
    comm_latency = 0
    latency_total = 0
    comp_requirement = 0
    # comm_Kbits = 0
 

    net_graph, inference_graph = get_graph(net_detail, model_details)
    num_net_node, network_dim = net_graph.x.shape[0], net_graph.x.shape[1] # 网络节点特征的数量，目前是1
    num_inference_node, inference_dim = inference_graph.x.shape[0], inference_graph.x.shape[1] # 推理模型的分割点数和点特征的数量，分割点也是动作空间，目前分割点为8个，因为还要考虑输入层和输出层，点特征数为1

    # 创建计算时延和通信时延的矩阵
    comp_latency_matrix = np.zeros((num_net_node, 1))
    comm_latency_matrix = np.zeros((num_net_node, 1))

    # print("分割方案为：", actor_allocation_scheme)
    for i in range(num_net_node): # 这里输入的模型分割的策略，策略里的元素是代表第几个分割点，而不是下标从0开始的索引。
        # print('第 %d 个节点'%(i+1))
        comp_ability = net_graph.x[i, 0] # 节点的算力，取x中第一列的第i个元素
        if i == 0:
            comm_speed = net_graph.edge_attr[i, 0]
            if actor_allocation_scheme[i] == 0:
                comp_requirement = 0
                comm_Kbits = inference_graph.edge_attr[0,0]
            elif 0< actor_allocation_scheme[i] < num_inference_node: # 如果非第一节点i承担计算量，则计算前actor_allocation_scheme[i]个节点的计算量，和这个几点的输出量，再把分割点及前面清零。

                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-1] # 第几个节点索引下标时要减1
                inference_graph.x[0:actor_allocation_scheme[i], 0] = 0 # 给选过的点置零
            elif actor_allocation_scheme[i] == num_inference_node: # 如果分割点为最后一个，则计算之前的计算量，输出为最后一个分割点的输出，但是最后一个分割点没有自己的输出，它输出结果，所以输出量跟上一个分割点相同。
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-2] # 传输量这里要-2，因为边本身就比节点少一个，最后一个节点-1之后索引小标还是超过了，所以要-2、

        elif 0 < i < (num_net_node-1):  # i既不是第一个，也不是最后一个节点时
            comm_speed = net_graph.edge_attr[i, 0]
            if actor_allocation_scheme[i] == 0: # 如果非第一节点不承担计算量，则传输数据量与上次一样，起转发作用。
                comp_requirement = 0
                comm_Kbits = comm_Kbits
            elif 0< actor_allocation_scheme[i] < num_inference_node: # 如果非第一节点i承担计算量，则计算前actor_allocation_scheme[i]个节点的计算量，和这个几点的输出量，再把分割点及前面清零。
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-1] # 第几个节点索引下标时要减1

                inference_graph.x[0:actor_allocation_scheme[i], 0] = 0 # 给选过的点置零
            elif actor_allocation_scheme[i] == num_inference_node: # 如果分割点为最后一个，则计算之前的计算量，输出为最后一个分割点的输出，但是最后一个分割点没有自己的输出，它输出结果，所以输出量跟上一个分割点相同。
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                comm_Kbits = inference_graph.edge_attr[actor_allocation_scheme[i]-2] # 传输量这里要-2，因为边本身就比节点少一个，最后一个节点-1之后索引小标还是超过了，所以要-2、
            if type(comm_Kbits) is not float : comm_Kbits = comm_Kbits.item()
        elif i == (num_net_node-1): # 如果i是最后一个节点，那他没有传输量，也没有传输速度，但是可以承担计算量。 但是不需要给前面的计算量清0
            comm_speed = 0

            if actor_allocation_scheme[i] == 0: # 如果最后一个节点不承担计算量，则传输数据量为0
                comp_requirement = 0
                comm_Kbits = 0 # 最后一个节点的传输量永远是0
            # elif 0< actor_allocation_scheme[i] < num_inference_node: # 如果最后一个节点i承担计算量，则计算前actor_allocation_scheme[i]个节点的计算量
            elif actor_allocation_scheme[i] != 0: # 如果最后一个节点i承担计算量，则计算前actor_allocation_scheme[i]个节点的计算量
                comp_requirement = sum(inference_graph.x[0:actor_allocation_scheme[i],0])
                # inference_graph.x[0:actor_allocation_scheme[i], 0] = 0 # 给选过的点置零

                comm_Kbits = 0 # 最后一个节点的传输量永远是0

        
        # comp_latency[i] = comp_requirement/comp_ability
        # comm_latency[i] = comm_Kbits / (comm_speed+1e-10)
        comp_latency_now = comp_requirement/comp_ability
        comm_latency_now = comm_Kbits/(comm_speed+1e-10)
        latency_now = comp_latency_now+comm_latency_now # !当前节点的延迟为计算延迟和通信延迟之和

        comp_latency += comp_latency_now # comm_latency单位是ms，这里节点的算力是GFlops，但是计算量为MFLOPs，正常来说应该给算力*1000才能得到秒，但是我们要的是毫秒，所以不给算力换算了。
        comm_latency += comm_latency_now # comm_latency单位是ms，这里传输量为KB，但是计算量为MBps，正常来说应该给comm_Kbits除以1000才能得到秒，但是我们要的是毫秒，所以不给comm_Kbits换算了。

        # 将计算时延和通信时延存储到矩阵中
        comp_latency_matrix[i] = comp_latency_now
        comm_latency_matrix[i] = comm_latency_now

        # print('当前 %d 节点的延迟为 %f ms, 其中计算延迟: %f ms, 计算量: %f MFLOPs, 算力: %f GFlops。传输延迟: %f ms, 传输量: %f KBits, 传输速度: %f MBps'%(i+1, latency_now, comp_latency_now, comp_requirement, comp_ability, comm_latency_now, comm_Kbits, comm_speed))

    # print(comp_latency, comm_latency)
    # latency_total = np.sum(comp_latency) + np.sum(comm_latency)
    latency_total = comp_latency+comm_latency
    # print('总延迟：%f ms, 计算延迟：%f ms, 通信延迟：%f ms。'%(latency_total, comp_latency, comm_latency))
    return latency_total, comp_latency, comm_latency, comp_latency_matrix, comm_latency_matrix

# if __name__ == '__main__':
    
#     pass