import torch


def precompute_rope(head_dim, max_seq_len, theta=10000.0):

    # Frequencies for each pair of dimensions
    freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        )
    )

    # Token positions: 0, 1, 2, ... max_seq_len-1
    positions = torch.arange(max_seq_len)

    # Position × frequency
    angles = torch.outer(positions, freq)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):

    # Separate dimension pairs
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Rotate each pair
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Put the pairs back together
    rotated = torch.stack(
        (rotated_even, rotated_odd),
        dim=-1
    )

    return rotated.flatten(-2)

if __name__ == "__main__":

    head_dim = 4
    max_seq_len = 3

    cos, sin = precompute_rope(
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )

    print("Cos:")
    print(cos)

    print("\nSin:")
    print(sin)

    # One vector at each of 3 token positions
    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0]
    ])

    print("\nBefore RoPE:")
    print(x)

    output = apply_rope(x, cos, sin)

    print("\nAfter RoPE:")
    print(output)