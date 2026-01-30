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


class CrossAttention(nn.Module):
    """Cross Attention Module for multi-modal processing"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value):
        """
        Args:
            query: [B, S, D] - Query sequence (e.g., text)
            key_value: [B, S, D] - Key/Value sequence (e.g., audio)
        Returns:
            Output of cross attention [B, S, D]
        """
        # Cross attention: query attends to key_value
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        output = query + self.dropout(attn_output)  # Residual connection
        return self.norm(output)


class StateEncoder(nn.Module):
    """State Encoder for extracting kinematic features from previous action"""
    def __init__(self, action_dim, state_dim=256, embed_dim=1024):
        super().__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        # Linear projection from action dimension to state dimension
        self.state_proj = nn.Linear(action_dim, state_dim)
        # Further project state to embed_dim
        self.embed_proj = nn.Linear(state_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, prev_action):
        """
        Args:
            prev_action: [B, action_dim] - Previous action primitive
        Returns:
            Embedded state representation [B, embed_dim]
        """
        # Project action to state space
        state_features = F.relu(self.state_proj(prev_action))
        # Project to embedding dimension
        embedded_state = self.embed_proj(state_features)
        return self.norm(self.dropout(embedded_state))


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
        
        # Cross attention module for multimodal fusion
        self.cross_attention = CrossAttention(embed_dim, num_heads)
        
        # State encoder for previous action feedback
        self.state_encoder = StateEncoder(action_dim, state_dim=256, embed_dim=embed_dim)
        
        # Projection layer to generate latent motion query
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        
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

    def forward(self, text, audio, prev_action=None):
        """
        Forward pass with recurrent motion state feedback (RMSF)
        Args:
            text: [B, S, text_dim] - Text features
            audio: [B, S, audio_dim] - Audio features  
            prev_action: [B, action_dim] - Previous action primitive (optional)
        Returns:
            action_prediction: [B, action_dim] - Predicted action
            latent_query: [B, embed_dim] - Latent motion query for executor
        """
        # Embed text and audio
        text_proj = self.text_embedding(text)  # [B, S, D]
        audio_proj = self.audio_embedding(audio)  # [B, S, D]
        
        # Multi-modal fusion using cross attention
        if self.use_multimodal:
            # Text attends to audio (or vice versa)
            fused_features = self.cross_attention(text_proj, audio_proj)
        else:
            fused_features = text_proj
        
        # Add positional encoding
        x = self.pos_encoder(fused_features)
        
        # Inject previous action state if provided (RMSF mechanism)
        if prev_action is not None:
            # Encode the previous action state
            prev_state_embed = self.state_encoder(prev_action)  # [B, embed_dim]
            # Expand to match sequence length
            prev_state_seq = prev_state_embed.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, S, embed_dim]
            # Concatenate with input features
            x = torch.cat([x, prev_state_seq], dim=-1)  # [B, S, 2*embed_dim]
            # Project back to original dimension
            x = nn.Linear(x.size(-1), self.text_embedding.out_features).to(x.device)(x)
        
        # Generate auto-regressive mask
        seq_len = x.size(1)
        # mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # Pass through all transformer layers
        for layer in self.layers:
            x = layer(x, mask=None)
        
        # Get final hidden state (take last time step)
        final_hidden = x[:, -1, :]  # [B, embed_dim]
        
        # Generate latent motion query
        latent_query = self.query_projection(final_hidden)  # [B, embed_dim]
        
        # Predict action
        action_prediction = self.fc_out(final_hidden)  # [B, action_dim]
        
        return action_prediction, latent_query


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
    def __init__(self, text_features, audio_features, action_features, window_size):
        self.text_features = text_features.astype(np.float32)
        self.audio_features = audio_features.astype(np.float32)
        self.action_features = action_features.astype(np.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.text_features) - self.window_size

    def __getitem__(self, idx):
        text_feature = self.text_features[idx:idx + self.window_size]
        audio_feature = self.audio_features[idx:idx + self.window_size]
        action_feature = self.action_features[idx + self.window_size-1]
        # Return previous action if available
        prev_action = self.action_features[idx + self.window_size-2] if idx + self.window_size-2 >= 0 else np.zeros_like(action_feature)
        return {
            'text': text_feature,
            'audio': audio_feature,
            'action': action_feature,
            'prev_action': prev_action
        }