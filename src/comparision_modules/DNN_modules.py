"""
Project: TaCo
File: DNN_modules.py
Description: This module defines the DNN modules used in the system. "Refer to Meng et al. 2023 - Task offloading optimization mechanism based on deep neural network in edge-cloud environment"
However, the details of the model are not provided, thus we build the model based on other provided code.

Author:  Lance
Created: 2024-12-25
Email: lance.lz.kong@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskOffloadingDNN(nn.Module):
    def __init__(self):
        super(TaskOffloadingDNN, self).__init__()
        # Define hidden layers
        self.fc1 = nn.Linear(44, 128)  # Input is [44] -> [128]
        self.fc2 = nn.Linear(128, 256)  # [128] -> [256]
        self.fc = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)  # [256] -> [128]
        self.fc4 = nn.Linear(128, 120)  # [128] -> [120], ensure it can map to 15 samples each with 8 categories

        # Final layer, output shape should be [15, 8], corresponding to 15 samples and 8 categories each
        self.fc5 = nn.Linear(120, 120)  # Optional hidden layer, output 120
        self.fc6 = nn.Linear(120, 15 * 8)  # Final layer, output dimension of 15 * 8 = 120

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x input shape is [batch_size=1, 44], i.e., [1, 44]
        
        x = F.relu(self.fc1(x))  # Output shape [1, 128]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # Output shape [1, 256]
        # x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc3(x))  # Output shape [1, 128]
        x = self.dropout(x)
        x = F.relu(self.fc4(x))  # Output shape [1, 120]
        x = self.dropout(x)

        x = F.relu(self.fc5(x))  # Output shape [1, 120]
        x = self.fc6(x)  # Output shape [1, 120]

        # Reshape to [15, 8], i.e., 15 samples and 8 categories each
        x = x.view(15, 8)  # Reshape output to [15, 8]
        
        # Apply log_softmax activation function
        x = F.log_softmax(x, dim=1)  # Apply softmax to each category (dimension 1)
        
        return x

# # 定义DNN模型
# class TaskOffloadingDNN(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(TaskOffloadingDNN, self).__init__()
#         layers = []
#         for i in range(len(hidden_sizes)):
#             if i == 0:
#                 layers.append(nn.Linear(input_size, hidden_sizes[i]))
#             else:
#                 layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(hidden_sizes[-1], output_size))
#         layers.append(nn.Softmax(dim=1))  # 输出概率，用于二分类
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

#*===========测试成功
# if __name__ == '__main__':
#     # 参数设置
#     input_size = 10  # 输入特征维度
#     hidden_sizes = [64, 128, 64]  # 隐藏层节点数
#     output_size = 1  # 输出卸载策略
#     learning_rate = 0.01
#     batch_size = 64
#     num_epochs = 100

#     # 创建模型和优化器
#     model = TaskOffloadingDNN(input_size, hidden_sizes, output_size)
#     criterion = nn.BCELoss()  # 二分类交叉熵损失
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # 模拟数据集
#     torch.manual_seed(42)
#     num_samples = 1000
#     X = torch.rand(num_samples, input_size)
#     y = torch.randint(0, 2, (num_samples, 1)).float()  # 0或1

#     # 数据加载
#     dataset = torch.utils.data.TensorDataset(X, y)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # 模型训练
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch_X, batch_y in data_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")

#     # 模型评估
#     model.eval()
#     with torch.no_grad():
#         test_X = torch.rand(100, input_size)
#         test_y = torch.randint(0, 2, (100, 1)).float()
#         predictions = model(test_X)
#         predictions = (predictions > 0.5).float()
#         accuracy = (predictions == test_y).float().mean()
#         print(f"Test Accuracy: {accuracy:.2%}")
