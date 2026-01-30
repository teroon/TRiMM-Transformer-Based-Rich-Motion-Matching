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
dataloader = DataLoader(dataset, batch_size=64, shuffle=True,drop_last=True)  # Changed shuffle to True for better training

model.to(device)
# 加载本地模型
model_path = 'Data/transformer-5-8-4.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except:
    print("Warning: Could not load pre-trained model, starting from scratch.")
#model_after.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

gamma = (1e-5 / 1e-4) ** (1 / 3000)
scheduler = ExponentialLR(optimizer, gamma=gamma)

# Define loss functions for the new training objectives
def reconstruction_loss(predicted_query, ground_truth_action):
    """
    Reconstruction Loss: L_rec = 1 - (q_t · e_t^GT) / (||q_t|| ||e_t^GT||)
    Minimizes cosine distance between predicted latent query and ground truth motion embedding
    """
    # Normalize both tensors
    pred_norm = F.normalize(predicted_query, p=2, dim=-1)
    gt_norm = F.normalize(ground_truth_action, p=2, dim=-1)
    
    # Calculate cosine similarity
    cos_sim = torch.sum(pred_norm * gt_norm, dim=-1)
    
    # Reconstruction loss is 1 - cosine similarity
    rec_loss = 1 - cos_sim
    
    return rec_loss.mean()

def consistency_loss(predicted_query, ground_truth_action, prev_predicted_query=None, prev_ground_truth=None):
    """
    Consistency Loss: L_con = ||(q_t - q_{t-1}) - (e_t^GT - e_{t-1}^GT)||_2^2
    Penalizes abrupt changes in predicted query velocity
    """
    if prev_predicted_query is not None and prev_ground_truth is not None:
        # Calculate velocity differences
        pred_velocity = predicted_query - prev_predicted_query
        gt_velocity = ground_truth_action - prev_ground_truth
        
        # Calculate consistency loss
        con_loss = F.mse_loss(pred_velocity, gt_velocity)
    else:
        # If no previous states, return 0
        con_loss = torch.tensor(0.0, device=predicted_query.device)
    
    return con_loss

searcher = Searcher.Searcher(
            "./Data/"
        )


text_first_eight_frames = text_features[:8]
audio_first_eight_frames = audio_features[:8]
action_first_frame = action_features[7]  

# 转换为 PyTorch 张量并添加批次维度
text_tensor = torch.tensor(text_first_eight_frames, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 8, 768]
audio_tensor = torch.tensor(audio_first_eight_frames, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 8, 2048]
prev_action_tensor = torch.tensor(action_features[6], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 750] 前一帧动作

# 将模型设置为评估模式
model.eval()

# 进行推理 - 使用RMSF机制（提供前一帧动作作为反馈）
with torch.no_grad():
    predicted_action, latent_query = model(text_tensor, audio_tensor, prev_action_tensor)  # Updated to return both outputs

# Ground truth action corresponding to the target frame
selected_vector_tensor = torch.from_numpy(action_first_frame).float().to(device)

# 实例化 MSELoss
mse_loss = torch.nn.MSELoss()
# 计算动作预测的MSE损失
action_mse_loss = mse_loss(predicted_action, selected_vector_tensor)
print(f"Initial test MSE Loss: {action_mse_loss.item():.6f}")

# Also calculate the reconstruction loss for the test
rec_test_loss = reconstruction_loss(latent_query, selected_vector_tensor)
print(f"Initial test Reconstruction Loss: {rec_test_loss.item():.6f}")


def train(model, dataloader, epochs, scheduler):
    model.train()
    
    # Loss weights as specified in the paper
    lambda_rec = 1.0
    lambda_con = 0.5
    
    for epoch in range(epochs):
        total_loss = 0.0
        rec_loss_total = 0.0
        con_loss_total = 0.0
        
        # Initialize previous states for consistency calculation
        prev_latent_query = None
        prev_gt_action = None
        
        for batch_idx, batch in enumerate(dataloader):
            # 直接加载整个 batch 到 GPU
            text = batch['text'].to(device)        # [batch_size, seq_len, text_dim]
            audio = batch['audio'].to(device)      # [batch_size, seq_len, audio_dim]
            next_action = batch['action'].to(device)  # [batch_size, action_dim]
            prev_action = batch['prev_action'].to(device)  # [batch_size, action_dim]
            
            optimizer.zero_grad()
            
            # Forward pass with previous action feedback (RMSF mechanism)
            predicted_action, latent_query = model(text, audio, prev_action)
            
            # Calculate reconstruction loss
            rec_loss = reconstruction_loss(latent_query, next_action)
            
            # Calculate consistency loss using the actual previous states from the batch
            con_loss = consistency_loss(latent_query, next_action, prev_latent_query, prev_gt_action)
            
            # Update previous states for next iteration
            prev_latent_query = latent_query.detach()  # Detach to prevent backprop through time
            prev_gt_action = next_action.detach()
            
            # Total loss as weighted sum
            total_batch_loss = lambda_rec * rec_loss + lambda_con * con_loss
            
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            rec_loss_total += rec_loss.item()
            con_loss_total += con_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_rec_loss = rec_loss_total / len(dataloader)
        avg_con_loss = con_loss_total / len(dataloader)
        
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}, Con Loss: {avg_con_loss:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        # 更新学习率
        scheduler.step()

train(model, dataloader, epochs=5000, scheduler=scheduler)