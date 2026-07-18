from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)

print(tokenizer.tokenize("elephant"))
print(len(tokenizer.get_vocab()))