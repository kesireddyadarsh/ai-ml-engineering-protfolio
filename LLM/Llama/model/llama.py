import torch
import torch.nn as nn

from config import ModelConfig
from model.rmsnorm import RMSNorm
from model.block import TransformerBlock


class TinyLlama(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 1. Token embedding
        # Token ID -> 256-dimensional vector
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.dim
        )

        # 2. Stack transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                hidden_dim=config.hidden_dim,
                max_seq_len=config.max_seq_len
            )
            for _ in range(config.n_layers)
        ])

        # 3. Final RMSNorm
        self.norm = RMSNorm(config.dim)

        # 4. Convert hidden representation
        # into vocabulary logits
        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False
        )

    def forward(self, tokens):

        # tokens:
        # [batch, sequence]
        #
        # Example:
        # [2, 10]

        # Token IDs -> embeddings
        # [B, S] -> [B, S, 256]
        x = self.token_embedding(tokens)

        # Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Convert to vocabulary logits
        # [B, S, 256] -> [B, S, 1000]
        logits = self.output(x)

        return logits


if __name__ == "__main__":

    config = ModelConfig()

    model = TinyLlama(config)

    # Fake token IDs for testing
    tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 10)
    )

    logits = model(tokens)

    print("Input token shape:")
    print(tokens.shape)

    print("\nLogits shape:")
    print(logits.shape)

    # Count model parameters
    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTotal parameters:")
    print(f"{total_parameters:,}")