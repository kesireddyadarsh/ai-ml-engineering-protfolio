import torch
import sentencepiece as spm

from config import ModelConfig
from model.llama import TinyLlama


# ----------------------------------------
# Device
# ----------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ----------------------------------------
# Load config
# ----------------------------------------

config = ModelConfig()


# ----------------------------------------
# Load tokenizer
# ----------------------------------------

tokenizer = spm.SentencePieceProcessor(
    model_file="tokenizer/tiny_llama_tokenizer.model"
)


# ----------------------------------------
# Load model
# ----------------------------------------

model = TinyLlama(config).to(device)

model.load_state_dict(
    torch.load(
        "tiny_llama_model.pt",
        map_location=device
    )
)

model.eval()


# ----------------------------------------
# Generate
# ----------------------------------------

def generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8
):

    token_ids = tokenizer.encode(
        prompt,
        out_type=int
    )

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only tokens that fit context window
            input_tokens = tokens[:, -config.max_seq_len:]

            logits = model(input_tokens)

            # Only use prediction from final position
            next_token_logits = logits[:, -1, :]

            # Temperature
            next_token_logits = (
                next_token_logits / temperature
            )

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

            # Stop at EOS
            if next_token.item() == tokenizer.eos_id():
                break

    output_text = tokenizer.decode(
        tokens[0].tolist()
    )

    return output_text


# ----------------------------------------
# Test
# ----------------------------------------

prompt = "Machine learning is"

output = generate(
    prompt,
    max_new_tokens=50
)

print("\nPrompt:")
print(prompt)

print("\nGenerated text:")
print(output)