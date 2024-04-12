import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate=0.1):  # Include dropout_rate parameter
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Add a dropout layer after activation
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)  # Optionally, add dropout after the second linear layer too
        )

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_pool = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        batch_size = x.shape[0]
        query = self.query.repeat(batch_size, 1, 1)
        
        # Transpose x and query to fit [sequence_length, batch_size, feature_dim]
        x_transposed = x.transpose(0, 1)  # Change to [64, 100, 64]
        query_transposed = query.transpose(0, 1)  # Change to [1, 100, 64]

        # Apply attention
        x_pooled, _ = self.attention_pool(query_transposed, x_transposed, x_transposed)
        x_pooled = x_pooled.transpose(0, 1)  # Transpose back to [batch_size, sequence_length, feature_dim]

        # Since we are pooling, we generally want a single vector per item in the batch
        return x_pooled.mean(dim=1)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64,  dropout_rate=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer for the attention

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout_rate=dropout_rate),
                FeedForward(dim, mlp_dim, dropout_rate=dropout_rate)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout_rate=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate=dropout_rate)

        self.attention_pooling = AttentionPooling(dim, heads)
        self.to_latent = nn.Identity()

        self.linear_head = nn.Sequential(
            nn.Dropout(dropout_rate),  # Additional dropout before the classifier
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        # print(f"Shape before attention pooling: {x.shape}")  # Debug output
        x = self.attention_pooling(x)

        x = self.to_latent(x)
        return self.linear_head(x)