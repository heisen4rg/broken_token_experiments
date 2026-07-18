from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

print(tokenizer.tokenize("elephant"))

print(tokenizer.get_vocab()["ele"])

print(len(tokenizer.get_vocab()))