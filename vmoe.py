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
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)




class MoELayer(nn.Module):
    def __init__(self, num_experts, in_dim, expert_dim):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(in_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, in_dim)
        ) for _ in range(num_experts)])
        self.gating_network = nn.Sequential(
            nn.Linear(in_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Get expert predictions for each feature channel
        gating_weights = self.gating_network(x)  # (B, P, C, E)  B: Batch size, P: Patches, C: Channels, E: Experts

        # Reshape gating weights to match the shape of expert_output for element-wise multiplication
        gating_weights = gating_weights.unsqueeze(2)  # Add a singleton dimension at index 2 (channels)

        # Apply gating weights and expert predictions
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # (B, P, C, in_dim)
            # Perform element-wise multiplication after broadcasting gating_weights
            expert_outputs.append(expert_output * gating_weights)  

        # Combine expert outputs (weighted sum)
        x = torch.stack(expert_outputs, dim=-1).sum(dim=-1)  # (B, P, C, in_dim)

        return x



class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, num_experts=4):
        super().__init__()

        assert dim % heads == 0, "Input dimension q'dim' must be divisible by the number of heads 'heads'."

        # ... existing code ...
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        # ... 

        self.to_patch_embedding = nn.Sequential(
            # ... existing code ...
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            # ... 
        )
        self.pos_embedding = posemb_sincos_2d(
            # ... existing code ...
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
            # ... 
        )

        self.transformer = torch.nn.Transformer(dim, depth, heads, dim_head, mlp_dim)

        # MoE Layer
        self.moe_layer = MoELayer(num_experts, dim, expert_dim=mlp_dim)  # Define MoE layer

        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    # ... existing forward method ...

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        # Pass through MoE layer
        x = self.moe_layer(x)

        x = self.transformer(x)
        # ... rest of the forward method ...
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        return self.linear_head(x)
        # ... 
