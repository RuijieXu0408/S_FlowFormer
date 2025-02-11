import torch
import torch.nn as nn
from torch import einsum


class BroadMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(BroadMultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim/heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        # Remove usage of einops for JIT compiliation and performance improvement
        # Q = rearrange(Q.squeeze(), 'i (heads d) -> heads i d', heads=self.heads)
        Q = Q.squeeze()
        Q_i, Q_heads_d = Q.shape
        Q_d = Q_heads_d // self.heads
        Q = Q.view(Q_i, self.heads, Q_d).permute(1, 0, 2)
        
        # Remove usage of einops for JIT compiliation and performance improvement
        # K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)
        K_b, K_j, K_heads_d = K.shape
        K_d = K_heads_d // self.heads
        K = K.view(K_b, K_j, self.heads, K_d).permute(0, 2, 1, 3)

        dots = einsum('hid, bhjd -> bhij', Q, K) * self.scale # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, _, _ = K.shape
        _, N, _ = Q.shape

        # V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)
        V_b, V_j, V_heads_d = V.shape
        V_d = V_heads_d // self.heads
        V = V.view(V_b, V_j, self.heads, V_d).permute(0, 2, 1, 3)

        out = einsum('bhij, bhjd -> bhid', attn, V)
        # out = rearrange(out, 'b heads n d -> b n (heads d)', b=B, n=N)
        out_b, out_heads, out_n, out_d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(out_b, out_n, out_heads * out_d)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim/heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        # Q = rearrange(Q, 'b i (heads d) -> b heads i d', heads=self.heads)
        Q_b, Q_i, Q_heads_d = Q.shape
        Q_d = Q_heads_d // self.heads
        Q = Q.view(Q_b, Q_i, self.heads, Q_d).permute(0, 2, 1, 3)
        
        # K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)
        K_b, K_j, K_heads_d = K.shape
        K_d = K_heads_d // self.heads
        K = K.view(K_b, K_j, self.heads, K_d).permute(0, 2, 1, 3)

        dots = einsum('bhid, bhjd -> bhij', Q, K) * self.scale # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        attn = self.attend_with_rpe(Q, K)
        B, HW, _ = Q.shape

        # V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)
        V_b, V_j, V_heads_d = V.shape
        V_d = V_heads_d // self.heads
        V = V.view(V_b, V_j, self.heads, V_d).permute(0, 2, 1, 3)

        out = einsum('bhij, bhjd -> bhid', attn, V)
        # out = rearrange(out, 'b heads hw d -> b hw (heads d)', b=B, hw=HW)
        out_b, out_heads, out_hw, out_d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(out_b, out_hw, out_heads * out_d)

        return out

def LinearPositionEmbeddingSine(x: torch.Tensor, dim: int = 128, NORMALIZE_FACOR: float =1/200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    freq_bands = torch.linspace(0, dim//4-1, dim//4, device=x.device, dtype=x.dtype)
    
    width: int = freq_bands.size(0)
    result = torch.empty((x.size(0), x.size(1), width * 4), device=x.device, dtype=x.dtype)
    result[..., width * 0 : width * 1] = x[..., -2:-1] * freq_bands
    result[..., width * 1 : width * 2] = x[..., -2:-1] * freq_bands
    result[..., width * 2 : width * 3] = x[..., -1:]   * freq_bands
    result[..., width * 3 : width * 4] = x[..., -1:]   * freq_bands
    
    result *= NORMALIZE_FACOR * torch.pi
    
    result[..., width * 0 : width * 1] = result[..., width * 0 : width * 1].sin_()
    result[..., width * 1 : width * 2] = result[..., width * 1 : width * 2].cos_()
    result[..., width * 2 : width * 3] = result[..., width * 2 : width * 3].sin_()
    result[..., width * 3 : width * 4] = result[..., width * 3 : width * 4].cos_()
    return result
