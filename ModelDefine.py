import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math


class DiviedSpaceTimeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, use_time_attention):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # 修改空间注意力的维度处理
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.use_time_attention = use_time_attention
        # 新增空间特征变换
        self.spatial_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x shape: [B, S, D]
        if self.use_time_attention:
            # --- 时间注意力 ---
            x_time = x.permute(1, 0, 2)  # [S, B, D]
            attn_time, _ = self.time_attn(x_time, x_time, x_time, attn_mask=mask)
            x = x + attn_time.permute(1, 0, 2)  # 残差连接 [B, S, D]
            
            # --- 空间注意力 ---
            # 将特征维度视为通道，保持序列维度
            x_space = self.spatial_proj(x)  # [B, S, D]
            x_space = x_space.permute(1, 0, 2)  # [S, B, D]
            attn_space, _ = self.spatial_attn(x_space, x_space, x_space)
            x = x + attn_space.permute(1, 0, 2)  # 恢复形状 [B, S, D]
        else:
            x = x
        return self.norm(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
        # 将编码移动到指定的设备上
        if device is not None:
            self.encoding = self.encoding.to(device)

    def forward(self, x):
        # 确保编码和输入x在同一个设备上
        encoding = self.encoding[:, :x.size(1)].detach()
        if x.is_cuda:
            encoding = encoding.to(x.device)
        return x + encoding


class MultiModalGPT(nn.Module):
    def __init__(self, text_dim=768, audio_dim=2048, action_dim=750, 
                 embed_dim=1024, num_heads=8, num_layers=6, 
                 use_time_attention=True, use_multimodal=True):
        super().__init__()
        # 输入嵌入
        self.text_embedding = nn.Linear(text_dim, embed_dim)
        self.audio_embedding = nn.Linear(audio_dim, embed_dim)
        
        # 融合门控
        self.fusion_gate = nn.Linear(embed_dim*2, 2)  # 输入是concat后的维度
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, use_time_attention)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(embed_dim, action_dim)
        self.use_multimodal = use_multimodal

    def forward(self, text, audio):
        # 模态融合
        if self.use_multimodal:
            text_proj = self.text_embedding(text)  # [B, S, D]
            audio_proj = self.audio_embedding(audio)
            
            # 门控融合
            combined = torch.cat([text_proj, audio_proj], dim=-1)
            gate = torch.softmax(self.fusion_gate(combined), dim=-1)  # [B, S, 2]
            x = gate[..., 0:1] * text_proj + gate[..., 1:2] * audio_proj
        else:
            x = self.text_embedding(text)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 生成自回归掩码
        seq_len = x.size(1)
        #mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # 通过所有Transformer层
        for layer in self.layers:
            x = layer(x, mask=None)
        
        # 预测动作
        return self.fc_out(x[:, -1, :])  # 取最后一个时间步

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_time_attention):
        super().__init__()
        self.attention = DiviedSpaceTimeAttention(embed_dim, num_heads, use_time_attention)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # 注意力子层
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈子层
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

    
class MultiModalDataset(Dataset):
    def __init__(self, text_features, audio_features, action_features,window_size):
        self.text_features = text_features.astype(np.float32)
        self.audio_features = audio_features.astype(np.float32)
        self.action_features = action_features.astype(np.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.text_features) - self.window_size

    def __getitem__(self, idx):
        text_feature = self.text_features[idx:idx + self.window_size ]
        audio_feature = self.audio_features[idx:idx + self.window_size]
        action_feature = self.action_features[idx + self.window_size-1 ]
        return {
            'text': text_feature,
            'audio': audio_feature,
            'action': action_feature,
        }
    