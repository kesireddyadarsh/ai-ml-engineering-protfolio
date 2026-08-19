import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="/Users/adarsh/Documents/GitHub/Untitled/LLM/Llama/data/training.txt",
    model_prefix="tokenizer/tiny_llama_tokenizer",
    vocab_size=1000,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3
)

print("Tokenizer training complete.")