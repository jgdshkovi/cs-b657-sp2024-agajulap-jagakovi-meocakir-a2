import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
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
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ShuffleViT(nn.Module):
    def __init__(self, *, model_class, image_size, patch_size, num_classes, dim, depth, heads, mlp_ratio, channels=3,
                 dim_head=64, total_image_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        if model_class == 'Plain-Old-CIFAR10':
            self.lens_size = image_size
        if model_class == 'D-shuffletruffle':
            self.lens_size = 16
        elif model_class == 'N-shuffletruffle':
            self.lens_size = 8
        else:
            raise ValueError(f'Unknown model_class {model_class}')

        lens_image_height, lens_image_width = pair(self.lens_size)
        patch_height, patch_width = pair(patch_size)

        assert lens_image_height % patch_height == 0 and lens_image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=lens_image_height // patch_height,
            w=lens_image_width // patch_width,
            dim=dim,
        )
        mlp_dim = mlp_ratio * dim
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        lens_count = (image_size ** 2) // (self.lens_size ** 2)
        self.sequential = nn.Sequential(
            nn.Linear(in_features=lens_count, out_features=lens_count),
            nn.LeakyReLU(),
        )
        self.project = nn.Linear(in_features=lens_count, out_features=1, bias=True)
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device
        img = rearrange(img, 'B c (h s1) (w s2) -> (h w) B c s1 s2', s1=self.lens_size, s2=self.lens_size)
        L, B, _, _, _ = img.shape
        x_emb = torch.zeros(L, B, self.dim).to(device)

        for i, shuffled_patch in enumerate(img):
            x = self.to_patch_embedding(shuffled_patch)
            x += self.pos_embedding.to(device, dtype=x.dtype)
            x = self.transformer(x)
            x = x.mean(dim=1)
            x = self.to_latent(x)
            x_emb[i] = x

        x_emb = x_emb[torch.randperm(L)]
        x_emb = x_emb.permute(1, 2, 0)
        x_emb = self.sequential(x_emb)
        x_emb = self.project(x_emb)
        x_emb = x_emb.squeeze()
        return self.linear_head(x_emb)
