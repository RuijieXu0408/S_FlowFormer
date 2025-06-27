import torch
import torch.nn as nn
from torch import einsum


class BroadMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(BroadMultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim/heads) ** -0.5

    def forward(self, Q, K, V):
        _, Q_i, Q_heads_d = Q.shape
        Q_d = Q_heads_d // self.heads
        Q_h = Q.view(Q_i, self.heads, Q_d).permute(1, 0, 2)
        
        # Remove usage of einops for JIT compiliation and performance improvement
        # K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)
        K_b, K_j, K_heads_d = K.shape
        K_d = K_heads_d // self.heads
        K_h = K.view(K_b, K_j, self.heads, K_d).permute(0, 2, 1, 3)

        # V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)
        V_b, V_j, V_heads_d = V.shape
        V_d = V_heads_d // self.heads
        V = V.view(V_b, V_j, self.heads, V_d).permute(0, 2, 1, 3)

        dots = einsum('hid, bhjd -> bhij', Q_h, K_h) * self.scale # (b hw) heads 1 pointnum
        attn = dots.softmax(dim=-1)
        
        out  = einsum('bhij, bhjd -> bhid', attn, V)
        # out = rearrange(out, 'b heads n d -> b n (heads d)', b=B, n=N)
        out_b, out_heads, out_n, out_d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(out_b, out_n, out_heads * out_d)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # Q = rearrange(Q, 'b i (heads d) -> b heads i d', heads=self.heads)
        Q_b, Q_i, Q_heads_d = Q.shape
        Q_d = Q_heads_d // self.heads
        Q = Q.view(Q_b, Q_i, self.heads, Q_d).permute(0, 2, 1, 3)
        
        # K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)
        K_b, K_j, K_heads_d = K.shape
        K_d = K_heads_d // self.heads
        K = K.view(K_b, K_j, self.heads, K_d).permute(0, 2, 1, 3)

        dots = einsum('bhid, bhjd -> bhij', Q, K) * self.scale # (b hw) heads 1 pointnum
        attn = self.attend(dots)

        # V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)
        V_b, V_j, V_heads_d = V.shape
        V_d = V_heads_d // self.heads
        V = V.view(V_b, V_j, self.heads, V_d).permute(0, 2, 1, 3)

        out = einsum('bhij, bhjd -> bhid', attn, V)
        
        # out = rearrange(out, 'b heads hw d -> b hw (heads d)', b=B, hw=HW)
        out_b, out_heads, out_hw, out_d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(out_b, out_hw, out_heads * out_d)

        return out


def LinearPositionEmbeddingSine(
    x: torch.Tensor,
    dim: int = 128,
    NORMALIZE_FACTOR: float = 1/200
) -> torch.Tensor:
    """
    Same behavior as the original, but built via elementwise ops + concat
    (no in-place slice-assign => no ScatterND).
    
    Args:
      x: Tensor of shape [B, N, 2]
      dim: total embedding dimension (must be divisible by 4)
      NORMALIZE_FACTOR: scaling before sin/cos
    Returns:
      Tensor of shape [B, N, dim]
    """
    width = dim // 4
    freq_bands = torch.arange(width, device=x.device, dtype=x.dtype) * NORMALIZE_FACTOR * torch.pi
    
    x0 = x[..., -2:-1]    # shape [B, N, 1]
    x1 = x[..., -1:]      # shape [B, N, 1]

    arg0 = x0 * freq_bands 
    arg1 = x1 * freq_bands

    sin0 = torch.sin(arg0)
    cos0 = torch.cos(arg0)
    sin1 = torch.sin(arg1)
    cos1 = torch.cos(arg1)

    return torch.cat([sin0, cos0, sin1, cos1], dim=-1)
