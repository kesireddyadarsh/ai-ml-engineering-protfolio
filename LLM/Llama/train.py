import torch
import torch.nn.functional as F
import sentencepiece as spm

from config import ModelConfig
from model.llama import TinyLlama


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

config = ModelConfig()


# --------------------------------------------------
# 3. Load tokenizer
# --------------------------------------------------

tokenizer = spm.SentencePieceProcessor(
    model_file="tokenizer/tiny_llama_tokenizer.model"
)


# --------------------------------------------------
# 4. Load training text
# --------------------------------------------------

with open(
    "data/training.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

print("Characters in dataset:", len(text))


# --------------------------------------------------
# 5. Convert entire dataset to token IDs
# --------------------------------------------------

token_ids = tokenizer.encode(
    text,
    out_type=int
)

print("Tokens in dataset:", len(token_ids))

data = torch.tensor(
    token_ids,
    dtype=torch.long
)


# --------------------------------------------------
# 6. Create Tiny Llama
# --------------------------------------------------

model = TinyLlama(config).to(device)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Model parameters:", f"{total_parameters:,}")


# --------------------------------------------------
# 7. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# --------------------------------------------------
# 8. Training settings
# --------------------------------------------------

batch_size = 8
seq_len = config.max_seq_len
steps = 1000


# --------------------------------------------------
# 9. Create training batches
# --------------------------------------------------

def get_batch():

    # Random starting positions
    starts = torch.randint(
        0,
        len(data) - seq_len - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + seq_len]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + seq_len + 1]
        for i in starts
    ])

    return x.to(device), y.to(device)


# --------------------------------------------------
# 10. Training
# --------------------------------------------------

model.train()

for step in range(steps):

    x, y = get_batch()

    # Forward pass
    logits = model(x)

    # logits:
    # [batch, sequence, vocabulary]
    #
    # y:
    # [batch, sequence]

    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        y.reshape(-1)
    )

    # Remove gradients from previous step
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    if step % 50 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.4f}"
        )


# --------------------------------------------------
# 11. Save trained weights
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "tiny_llama_model.pt"
)

print("Training complete.")
print("Model saved as tiny_llama_model.pt")