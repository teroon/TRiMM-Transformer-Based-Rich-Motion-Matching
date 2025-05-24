import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ModelDefine
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ExponentialLR
import Searcher

def remove_nan(np_array):
    np_array = np.where(np.isnan(np_array) | np.isinf(np_array), 0, np_array)
    return np_array

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ModelDefine.MultiModalGPT()
#model_after = ModelDefine.ActionPredictionModel(embed_dim, action_dim, hidden_dim=512, num_layers=2)

text_features = np.nan_to_num(np.load("Training-Text-768.npy"))#[0:9143]
audio_features =np.nan_to_num( np.load("Training-Audio-2048.npy"))#[0:9143]
action_features= np.nan_to_num(np.load("Training-Action-750.npy"))/100#[0:9143]
max=np.max(action_features)
min=np.min(action_features)
#text_features = normalization(text_features)
#audio_features = normalization(audio_features)

dataset = ModelDefine.MultiModalDataset(text_features, audio_features,action_features, window_size=8)
test_indices = list(range(0, 1940))
test_dataset = Subset(dataset, test_indices)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False,drop_last=True)

model.to(device)
# 加载本地模型
model_path = 'Data/transformer-5-8-4.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
#model_after.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

gamma = (1e-5 / 1e-4) ** (1 / 3000)
scheduler = ExponentialLR(optimizer, gamma=gamma)

criterion_sequence = nn.MSELoss()
criterion_action = nn.MSELoss()
searcher = Searcher.Searcher(
            "./Data/"
        )

# 提取前八帧数据
text_first_eight_frames = text_features[:8]
audio_first_eight_frames = audio_features[:8]

# 转换为 PyTorch 张量并添加批次维度
text_tensor = torch.tensor(text_first_eight_frames, dtype=torch.float32).unsqueeze(0).to(device)
audio_tensor = torch.tensor(audio_first_eight_frames, dtype=torch.float32).unsqueeze(0).to(device)

# 将模型设置为评估模式
model.eval()

# 进行推理
with torch.no_grad():
    predicted_action = model(text_tensor, audio_tensor).cpu()
selected_vector_tensor = torch.from_numpy(action_features[7]).float()
# 实例化 MSELoss
mse_loss = torch.nn.MSELoss()
# 计算损失
loss = mse_loss(predicted_action, selected_vector_tensor)
print(loss.item())


def train(model, dataloader, epochs, scheduler):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            # 直接加载整个 batch 到 GPU
            text = batch['text'].to(device)        # [64, 8, 768]
            audio = batch['audio'].to(device)      # [64, 8, 512]
            next_action = batch['action'].to(device)  # [64, 54]

            optimizer.zero_grad()
            # 模型输出预测动作
            predicted_action = model(text, audio)  # 期望输出 [64, 54]
            loss = criterion_action(predicted_action, next_action)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        # 更新学习率
        scheduler.step()

train(model, dataloader, epochs=5000, scheduler=scheduler)

