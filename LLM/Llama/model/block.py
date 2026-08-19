import torch
import torch.nn as nn

from model.rmsnorm import RMSNorm
from model.attention import CausalSelfAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        hidden_dim,
        max_seq_len
    ):
        super().__init__()

        # RMSNorm before attention
        self.attention_norm = RMSNorm(dim)

        # Causal self-attention
        self.attention = CausalSelfAttention(
            dim=dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len
        )

        # RMSNorm before feed-forward
        self.ffn_norm = RMSNorm(dim)

        # SwiGLU feed-forward network
        self.feedforward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward + residual connection
        x = x + self.feedforward(
            self.ffn_norm(x)
        )

        return x


if __name__ == "__main__":

    block = TransformerBlock(
        dim=256,
        n_heads=4,
        hidden_dim=768,
        max_seq_len=128
    )

    x = torch.randn(
        2,      # batch
        10,     # sequence
        256     # embedding dimension
    )

    output = block(x)

    print("Input shape:")
    print(x.shape)

    print("\nTransformer block output shape:")
    print(output.shape)