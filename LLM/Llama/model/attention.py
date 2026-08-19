import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import apply_rope, precompute_rope


class CausalSelfAttention(nn.Module):

    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

        # Query, Key, Value projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

        # Final output projection
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):

        batch_size, seq_len, dim = x.shape

        # 1. Create Query, Key and Value
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 2. Split into multiple attention heads
        # [B, S, D] -> [B, S, H, head_dim]
        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        )

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        )

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        )

        # 3. Move head dimension forward
        # [B, S, H, head_dim] -> [B, H, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Precompute RoPE values
        cos, sin = precompute_rope(
            self.head_dim,
            seq_len
        )

        # Move RoPE values to same device as input
        cos = cos.to(x.device)
        sin = sin.to(x.device)

        # [S, head_dim/2] -> [1, 1, S, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 5. Apply RoPE to Query and Key
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 6. Calculate attention scores
        # Q: [B, H, S, head_dim]
        # K.T: [B, H, head_dim, S]
        # scores: [B, H, S, S]
        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        # 7. Scale scores
        scores = scores / (self.head_dim ** 0.5)

        # 8. Create causal mask
        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        # Block future tokens
        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # 9. Convert scores into probabilities
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # 10. Multiply attention probabilities by Values
        # [B, H, S, S] x [B, H, S, head_dim]
        # -> [B, H, S, head_dim]
        output = torch.matmul(
            attention_weights,
            v
        )

        # 11. Combine all attention heads
        # [B, H, S, head_dim]
        # -> [B, S, H, head_dim]
        output = output.transpose(1, 2)

        # Make tensor contiguous before reshape
        output = output.contiguous()

        # [B, S, H, head_dim]
        # -> [B, S, D]
        output = output.view(
            batch_size,
            seq_len,
            dim
        )

        # 12. Final output projection
        output = self.wo(output)

        return output


if __name__ == "__main__":

    attention = CausalSelfAttention(
        dim=256,
        n_heads=4,
        max_seq_len=128
    )

    # batch = 2
    # sequence length = 10
    # embedding dimension = 256
    x = torch.randn(
        2,
        10,
        256
    )

    output = attention(x)

    print("Input shape:")
    print(x.shape)

    print("\nAttention output shape:")
    print(output.shape)