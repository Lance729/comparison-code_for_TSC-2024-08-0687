"""
Project: TaCo
File: lstm_based_s2s.py
Description: This module defines the LSTM-based Seq2Seq model for the task offloading optimization in edge-cloud environment. Refer to "Cai et al. 2024 - Dependency-Aware Task Scheduling for Vehicular Networks Enhanced by the Integration of Sensing, Communication and Computing".

Author:  Lance
Created: 2024-12-25
Email: lance.lz.kong@gmail.com
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 超参数
input_dim = 100    # 输入数据的维度
hidden_dim = 256   # LSTM的隐藏层维度
output_dim = 10    # 输出数据的维度
num_layers = 2     # LSTM的层数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


# 解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell


# 序列到序列模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        hidden, cell = self.encoder(encoder_inputs)
        outputs, _, _ = self.decoder(decoder_inputs, hidden, cell)
        return outputs


# 数据生成函数
def generate_data(num_samples, timesteps, input_dim, output_dim):
    encoder_input_data = np.random.random((num_samples, timesteps, input_dim))
    decoder_input_data = np.random.random((num_samples, timesteps, output_dim))
    decoder_target_data = np.random.random((num_samples, timesteps, output_dim))
    return (
        torch.tensor(encoder_input_data, dtype=torch.float32),
        torch.tensor(decoder_input_data, dtype=torch.float32),
        torch.tensor(decoder_target_data, dtype=torch.float32),
    )


# 模型训练函数
def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, epochs, batch_size, criterion, optimizer):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for i in range(0, len(encoder_input_data), batch_size):
            encoder_inputs = encoder_input_data[i:i+batch_size].to(device)
            decoder_inputs = decoder_input_data[i:i+batch_size].to(device)
            targets = decoder_target_data[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(encoder_inputs, decoder_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (len(encoder_input_data) // batch_size)}")


# 推理测试函数
def test_model(model, encoder_input, decoder_input, timesteps):
    model.eval()
    with torch.no_grad():
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)

        hidden, cell = model.encoder(encoder_input)
        outputs = []
        input_step = decoder_input[:, 0:1, :]  # 初始化解码器输入

        for t in range(timesteps):
            output, hidden, cell = model.decoder(input_step, hidden, cell)
            outputs.append(output)
            input_step = output  # 使用当前输出作为下一步的输入

        outputs = torch.cat(outputs, dim=1)
        return outputs.cpu().numpy()


# 主函数
def main():
    # 数据生成
    num_samples = 1000
    timesteps = 20
    encoder_input_data, decoder_input_data, decoder_target_data = generate_data(num_samples, timesteps, input_dim, output_dim)

    # 初始化模型
    encoder = Encoder(input_dim, hidden_dim, num_layers).to(device)
    decoder = Decoder(output_dim, hidden_dim, num_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 20
    batch_size = 64
    train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, epochs, batch_size, criterion, optimizer)

    # 测试模型
    test_encoder_input = encoder_input_data[:1]
    test_decoder_input = decoder_input_data[:1]
    predictions = test_model(model, test_encoder_input, test_decoder_input, timesteps)
    print("Predicted sequence:", predictions)


if __name__ == "__main__":
    main()

