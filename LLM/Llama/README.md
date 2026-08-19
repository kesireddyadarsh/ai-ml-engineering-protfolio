# Tiny Llama-Style Language Model From Scratch

A small **Llama-style decoder-only language model implemented and trained from scratch using PyTorch**.

The goal of this project is to understand the internal architecture and training process of modern Large Language Models rather than relying on a pretrained model.

The model contains approximately **3.9 million trainable parameters** and was trained locally on an **Apple M4 MacBook Air using PyTorch MPS acceleration**.

## Architecture

The model implements several core components used in Llama-style transformer architectures:

* SentencePiece BPE tokenization
* Token embeddings
* RMSNorm
* Rotary Positional Embeddings (RoPE)
* Multi-head causal self-attention
* Causal attention masking
* SwiGLU feed-forward network
* Residual connections
* Stacked transformer blocks
* Autoregressive next-token prediction

### Model Configuration

| Parameter                     |     Value |
| ----------------------------- | --------: |
| Parameters                    | 3,922,176 |
| Vocabulary Size               |     1,000 |
| Embedding Dimension           |       256 |
| Transformer Layers            |         4 |
| Attention Heads               |         4 |
| Head Dimension                |        64 |
| Feed-Forward Hidden Dimension |       768 |
| Context Length                |       128 |

## Model Pipeline

```text
Raw Text
   ↓
SentencePiece BPE Tokenizer
   ↓
Token IDs
   ↓
Token Embeddings
   ↓
┌────────────────────────────┐
│ RMSNorm                    │
│ Causal Self-Attention      │
│   ├── Query / Key / Value  │
│   ├── RoPE on Q and K      │
│   ├── Scaled Dot Product   │
│   └── Causal Mask          │
│ Residual Connection        │
│ RMSNorm                    │
│ SwiGLU Feed-Forward        │
│ Residual Connection        │
└────────────────────────────┘
           × 4
   ↓
Final RMSNorm
   ↓
Linear Projection
   ↓
Vocabulary Logits
   ↓
Softmax / Sampling
   ↓
Next Token
```

## Tokenizer

A custom **SentencePiece BPE tokenizer** is trained directly on the training corpus.

```text
Text:
Machine learning is a branch of artificial intelligence.

Tokens:
['▁Machine', '▁learning', '▁is', '▁a',
 '▁b', 'ran', 'ch', '▁of',
 '▁artificial', '▁intelligence', '.']
```

The tokenizer uses a vocabulary of **1,000 tokens**.

## Training

The model is trained from randomly initialized weights using **autoregressive next-token prediction**.

For a sequence:

```text
Machine → learning → is → a → branch
```

training examples conceptually become:

```text
Input:   Machine
Target:  learning

Input:   Machine learning
Target:  is

Input:   Machine learning is
Target:  a
```

The implementation uses:

* Cross-entropy loss
* AdamW optimizer
* Causal masking
* Backpropagation
* PyTorch MPS acceleration

The trained model weights are stored in:

```text
tiny_llama_model.pt
```

## Example Generation

After training:

```text
Prompt:
Machine learning is
```

Example generated output:

```text
Machine learning is a balances these competing and transformer-based
architectures to process textive equire development. Training a large
language model requires enormous computational resources. During
pretraining, the model repeatedly predict...
```

Because this model is intentionally small and trained on a limited corpus, the generated text may contain grammatical errors, repetition, and memorized patterns. The objective is to demonstrate the complete language-model training and inference pipeline rather than compete with production-scale LLMs.

## Project Structure

```text
Llama/
│
├── data/
│   └── training.txt
│
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── test_tokenizer.py
│   ├── tiny_llama_tokenizer.model
│   └── tiny_llama_tokenizer.vocab
│
├── model/
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py
│   ├── feedforward.py
│   ├── block.py
│   └── llama.py
│
├── config.py
├── train.py
├── generate.py
└── tiny_llama_model.pt
```

## Running the Project

Install the required packages:

```bash
pip install torch sentencepiece
```

Train the tokenizer:

```bash
python tokenizer/train_tokenizer.py
```

Train the model:

```bash
python train.py
```

Run inference:

```bash
python generate.py
```

## Future Improvements

* Increase the size and diversity of the training corpus
* Add train/validation splits and validation loss
* Add checkpointing during training
* Implement configurable temperature and top-k/top-p sampling
* Add weight tying between embeddings and output projection
* Optimize RoPE computation
* Add KV caching for faster inference
* Train a larger model
* Add instruction/Q&A fine-tuning
* Add experiment tracking and evaluation metrics

## Tech Stack

**Python · PyTorch · SentencePiece · Apple Metal Performance Shaders (MPS)**


