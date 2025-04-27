import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import TransformerEncoderLayer

class DeiTBinary(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,  # Giảm embed_dim để nhẹ hơn
        depth=6,       # Giảm số layer
        n_heads=3,     # Giảm số head
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, 
                                   stride=patch_size)
        
        # CLS token và positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Head cho binary classification
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Khởi tạo weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H, W)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, n_patches, embed_dim)
        
        # Thêm CLS token và positional embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Classification head
        x = self.norm(x[:, 0])  # Lấy CLS token
        x = self.head(x)  # (B, 1) trong [0, 1]
        
        return x # (B,)