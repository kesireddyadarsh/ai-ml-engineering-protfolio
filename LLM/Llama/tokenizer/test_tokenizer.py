import sentencepiece as spm

sp = spm.SentencePieceProcessor(
    model_file="tokenizer/tiny_llama_tokenizer.model"
)

text = "Machine learning is a branch of artificial intelligence."

tokens = sp.encode(text, out_type=str)
token_ids = sp.encode(text, out_type=int)

print("Original text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

decoded = sp.decode(token_ids)

print("\nDecoded text:")
print(decoded)