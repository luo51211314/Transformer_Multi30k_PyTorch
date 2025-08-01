# Mytest.py
from transformers import AutoTokenizer

def check_token_ids(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = ['A', 'a', 'ing', 's', '', '</s>']
    
    print(f"\nModel: {model_name}")
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"'{token}': {token_id} (hex: {hex(token_id) if token_id else 'None'})")

if __name__ == "__main__":
    # 测试不同模型
    models = [
        "/root/autodl-tmp/models/opus-mt-de-en",
    ]
    for model in models:
        check_token_ids(model)