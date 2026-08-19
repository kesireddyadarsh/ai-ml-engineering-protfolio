import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )

        x = x / rms

        return self.weight * x


if __name__ == "__main__":

    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0]
    ])

    norm = RMSNorm(dim=4)

    output = norm(x)

    print("Input:")
    print(x)

    print("\nAfter RMSNorm:")
    print(output)