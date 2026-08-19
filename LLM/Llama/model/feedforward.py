import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()

        # 256 -> 768
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        # 768 -> 256
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        # 256 -> 768
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):

        # SwiGLU
        hidden = F.silu(self.w1(x)) * self.w3(x)

        # Project back to original dimension
        output = self.w2(hidden)

        return output


if __name__ == "__main__":

    feedforward = FeedForward(
        dim=256,
        hidden_dim=768
    )

    # batch=2, sequence=10, dimension=256
    x = torch.randn(2, 10, 256)

    output = feedforward(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)